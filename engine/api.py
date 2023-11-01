import cv2
import os
import json
import torch
import openai
import functools
import numpy as np
import face_detection
import io, tokenize
from math import ceil, floor
from augly.utils.base_paths import EMOJI_DIR
import augly.image as imaugs
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from transformers import (ViltProcessor, ViltForQuestionAnswering,
    OwlViTProcessor, OwlViTForObjectDetection,
    MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
    CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)
from diffusers import StableDiffusionInpaintPipeline
from .nms import nms
from engine.viper import vision_models
from engine.viper.vision_processes import make_fn
from torchvision import transforms
from typing import List, Union
from pynvml import nvmlDeviceGetHandleByIndex, nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetMemoryInfo, nvmlShutdown


def choose_gpu():
    nvmlInit()
    num_gpus = nvmlDeviceGetCount()
    memory_free = [nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free for i in range(num_gpus)]
    nvmlShutdown()
    gpu_number = memory_free.index(max(memory_free))
    print(f"gpu:{gpu_number}")
    # import pdb; pdb.set_trace()
    return gpu_number


def fix_boxes(boxes, max_x, max_y):
    for i, box in enumerate(boxes):
        boxes[i][0] = max(0, floor(box[0]))
        boxes[i][1] = max(0, floor(box[1]))
        boxes[i][2] = min(max_x, ceil(box[2]))
        boxes[i][3] = min(max_y, ceil(box[3]))
    return boxes


class API():
    class vqa_model():
        def __init__(self):
            print(f'Registering vqa model')
            self.device = f"cuda:{choose_gpu()}" if torch.cuda.is_available() else "cpu"
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
            self.model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-capfilt-large").to(self.device)
            self.model.eval()
        def predict(self,img,question):
            encoding = self.processor(img,question,return_tensors='pt')
            encoding = {k:v.to(self.device) for k,v in encoding.items()}
            with torch.no_grad():
                outputs = self.model.generate(**encoding)
            return self.processor.decode(outputs[0], skip_special_tokens=True)


    class loc_model():
        def __init__(self,):
            print(f'Registering loc model')
            self.device = f"cuda:{choose_gpu()}" if torch.cuda.is_available() else "cpu"
            self.processor = OwlViTProcessor.from_pretrained(
                "google/owlvit-large-patch14")
            self.model = OwlViTForObjectDetection.from_pretrained(
                "google/owlvit-large-patch14").to(self.device)
            self.model.eval()

        def normalize_coord(self,bbox,img_size):
            w,h = img_size
            x1,y1,x2,y2 = [int(v) for v in bbox]
            x1 = max(0,x1)
            y1 = max(0,y1)
            x2 = min(x2,w-1)
            y2 = min(y2,h-1)
            return [x1,y1,x2,y2]

        def predict(self,img,obj_name,thresh=0.1,nms_thresh=0.5):
            encoding = self.processor(
                text=[[f'a photo of {obj_name}']],
                images=img,
                return_tensors='pt')
            encoding = {k:v.to(self.device) for k,v in encoding.items()}
            with torch.no_grad():
                outputs = self.model(**encoding)
                for k,v in outputs.items():
                    if v is not None:
                        outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v

            target_sizes = torch.Tensor([img.size[::-1]])
            results = self.processor.post_process_object_detection(outputs=outputs,threshold=thresh,target_sizes=target_sizes)
            boxes, scores = results[0]["boxes"], results[0]["scores"]
            boxes = boxes.cpu().detach().numpy().tolist()
            scores = scores.cpu().detach().numpy().tolist()
            if len(boxes)==0:
                return []

            boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
            selected_boxes = []
            selected_scores = []
            for i in range(len(scores)):
                if scores[i] > thresh:
                    coord = self.normalize_coord(boxes[i],img.size)
                    selected_boxes.append(coord)
                    selected_scores.append(scores[i])

            selected_boxes, selected_scores = nms(
                selected_boxes,selected_scores,nms_thresh)
            return selected_boxes


    class object_detector_model():
        def __init__(self):
            print(f'Registering object_detector model')
            self.forward = make_fn(vision_models.ObjectDetector, 'object_detector', choose_gpu())

        def box_cxcywh_to_xyxy(self,x):
            x_c, y_c, w, h = x.unbind(1)
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return torch.stack(b, dim=1)

        def rescale_bboxes(self, out_bbox, size):
            img_w, img_h = size
            b = self.box_cxcywh_to_xyxy(out_bbox)
            b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
            return b

        def predict(self, image, threshold=0.7):
            transform = transforms.Compose([
                transforms.Resize(800),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            result = self.forward(transform(image))
            probas = result['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > threshold
            bboxes_scaled = self.rescale_bboxes(result['pred_boxes'][0, keep], image.size)
            boxes = bboxes_scaled.tolist()
            return fix_boxes(boxes, *image.size)


    class depth_model():
        def __init__(self):
            print(f'Registering depth model')
            self.forward = make_fn(vision_models.DepthEstimationModel, 'depth', choose_gpu())

        def predict(self, image):
            result = self.forward(transforms.ToTensor()(image))
            depth_map = result.transpose(0, 1).tolist()
            return depth_map


    class clip_model():
        def __init__(self):
            print(f'Registering clip model')
            self.forward = make_fn(vision_models.CLIPModel, 'clip', choose_gpu())

        def predict(self, image, prompt, task, return_index=True, negative_categories=None, return_scores=False):
            result = self.forward(transforms.ToTensor()(image), prompt, task, return_index, negative_categories, return_scores)
            return result


    class maskrcnn_model():
        def __init__(self):
            print(f'Registering maskrcnn model')
            self.forward = make_fn(vision_models.MaskRCNNModel, 'maskrcnn', choose_gpu())

        def predict(self, image, return_labels=False):
            result = self.forward(transforms.ToTensor()(image), return_labels)
            return result


    class owlvit_model():
        def __init__(self):
            print(f'Registering owlvit model')
            self.forward = make_fn(vision_models.OwlViTModel, 'owlvit', choose_gpu())

        def predict(self, image, text: List[str], return_labels: bool = False):
            result = self.forward(transforms.ToTensor()(image), text, return_labels)
            return result


    class glip_model():
        def __init__(self):
            print(f'Registering glip model')
            self.forward = make_fn(vision_models.GLIPModel, 'glip', choose_gpu())

        def predict(self, image, obj: Union[str, list], return_labels: bool = False, confidence_threshold=0.5):
            result = self.forward(transforms.ToTensor()(image), obj, return_labels, confidence_threshold)
            if isinstance(result, tuple):
                result = result[0]
            boxes = result.tolist()
            return boxes


    class tcl_model():
        def __init__(self):
            print(f'Registering tcl model')
            self.forward = make_fn(vision_models.TCLModel, 'tcl', choose_gpu())

        def predict(self, image, texts, task='classify', return_index=True):
            result = self.forward(transforms.ToTensor()(image), texts, task, return_index)
            return result


    class gpt3_model():
        def __init__(self):
            print(f'Registering gpt3 model')
            self.forward = make_fn(vision_models.GPT3Model, 'gpt3', choose_gpu())

        def predict(self, prompt, process_name='gpt3_qa'):
            result = self.forward(prompt, process_name)
            return result


    class codex_model():
        def __init__(self):
            print(f'Registering codex model')
            self.forward = make_fn(vision_models.CodexModel, 'codex', choose_gpu())

        def predict(self, prompt, input_type='image', prompt_file=None, base_prompt=None):
            result = self.forward(prompt, input_type, prompt_file, base_prompt)
            return result


    class blip_model():
        def __init__(self):
            print(f'Registering blip model')
            self.forward = make_fn(vision_models.BLIPModel, 'blip', choose_gpu())

        def predict(self, image, question=None, task='caption'):
            result = self.forward(image, question, task)
            return result


    class saliency_model():
        def __init__(self):
            print(f'Registering saliency model')
            self.forward = make_fn(vision_models.SaliencyModel, 'saliency', choose_gpu())

        def predict(self, image):
            result = self.forward(transforms.ToTensor()(image))
            return result


    class xvlm_model():
        def __init__(self):
            print(f'Registering xvlm model')
            self.forward = make_fn(vision_models.XVLMModel, 'xvlm', choose_gpu())

        def predict(self, image, text, negative_categories=None, task='binary'):
            result = self.forward(transforms.ToTensor()(image), text, task, negative_categories)
            return result


    class face_detection_model():
        def __init__(self):
            print(f'Registering face_detection model')
            self.model = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3, device=torch.device(f"cuda:{choose_gpu()}"))

        def predict(self, image):
            with torch.no_grad():
                faces = self.model.detect(np.array(image))
            return faces


    class segment_model():
        def __init__(self):
            print(f'Registering segment model')
            self.device = f"cuda:{choose_gpu()}" if torch.cuda.is_available() else "cpu"
            self.feature_extractor = MaskFormerFeatureExtractor.from_pretrained(
                "facebook/maskformer-swin-base-coco")
            self.model = MaskFormerForInstanceSegmentation.from_pretrained(
                "facebook/maskformer-swin-base-coco").to(self.device)
            self.model.eval()

        def predict(self, img):
            inputs = self.feature_extractor(images=img, return_tensors="pt")
            inputs = {k:v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            outputs = self.feature_extractor.post_process_panoptic_segmentation(outputs)[0]
            instance_map = outputs['segmentation'].cpu().numpy()
            objs = []
            print(outputs.keys())
            for seg in outputs['segments_info']:
                inst_id = seg['id']
                label_id = seg['label_id']
                category = self.model.config.id2label[label_id]
                mask = (instance_map==inst_id).astype(float)
                resized_mask = np.array(
                    Image.fromarray(mask).resize(
                        img.size,resample=Image.BILINEAR))
                Y,X = np.where(resized_mask>0.5)
                x1,x2 = np.min(X), np.max(X)
                y1,y2 = np.min(Y), np.max(Y)
                num_pixels = np.sum(mask)
                objs.append(dict(
                    mask=resized_mask,
                    category=category,
                    box=[x1,y1,x2,y2],
                    inst_id=inst_id
                ))

            return objs

    class select_model():
        def __init__(self):
            print(f'Registering select model')
            self.device = f"cuda:{choose_gpu()}" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14").to(self.device)
            self.model.eval()
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14")

        def calculate_sim(self,inputs):
            img_feats = self.model.get_image_features(inputs['pixel_values'])
            text_feats = self.model.get_text_features(inputs['input_ids'])
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
            return torch.matmul(img_feats,text_feats.t())

        def query_obj(self,query,objs,img):
            images = [img.crop(obj['box']) for obj in objs]
            text = [f'a photo of {q}' for q in query]
            inputs = self.processor(
                text=text, images=images, return_tensors="pt", padding=True)
            inputs = {k:v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad():
                scores = self.calculate_sim(inputs).cpu().numpy()

            obj_ids = scores.argmax(0)
            return [objs[i] for i in obj_ids]

    class replace_model():
        def __init__(self):
            print(f'Registering replace model')
            device = f"cuda:{choose_gpu()}" if torch.cuda.is_available() else "cpu"
            model_name = "runwayml/stable-diffusion-inpainting"
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_name,
                revision="fp16",
                torch_dtype=torch.float16)
            self.pipe = self.pipe.to(device)
            def dummy(images, **kwargs):
                return images, False
            self.pipe.safety_checker = dummy

        def predict(self, prompt, image, mask_image, guidance_scale=7.5, num_inference_steps=50):
            new_img = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                # strength=0.98,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps #200
            ).images[0]
            return new_img

    class query_obj_model():
        def __init__(self):
            print(f'Registering query_obj model')
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14").to(self.device)
            self.model.eval()
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        def calculate_sim(self,inputs):
            img_feats = self.model.get_image_features(inputs['pixel_values'])
            text_feats = self.model.get_text_features(inputs['input_ids'])
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
            return torch.matmul(img_feats,text_feats.t())

        def predict(self,query,objs,img):
            if len(objs)==0:
                images = [img]
                return []
            else:
                images = [img.crop(obj['box']) for obj in objs]

            if isinstance(query, str):
                query = [query]

            if len(query)==1:
                query = query + ['other']

            text = [f'a photo of {q}' for q in query]
            inputs = self.processor(
                text=text, images=images, return_tensors="pt", padding=True)
            inputs = {k:v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad():
                sim = self.calculate_sim(inputs)

            # if only one query then select the object with the highest score
            if len(query)==1:
                scores = sim.cpu().numpy()
                obj_ids = scores.argmax(0)
                obj = objs[obj_ids[0]]
                obj['class']=query[0]
                obj['class_score'] = 100.0*scores[obj_ids[0],0]
                return [obj]

            # assign the highest scoring class to each object but this may assign same class to multiple objects
            scores = sim.cpu().numpy()
            cat_ids = scores.argmax(1)
            for i,(obj,cat_id) in enumerate(zip(objs,cat_ids)):
                class_name = query[cat_id]
                class_score = scores[i,cat_id]
                obj['class'] = class_name #+ f'({score_str})'
                obj['class_score'] = round(class_score*100,1)

            # sort by class scores and then for each class take the highest scoring object
            objs = sorted(objs,key=lambda x: x['class_score'],reverse=True)
            objs = [obj for obj in objs if 'class' in obj]
            classes = set([obj['class'] for obj in objs])
            new_objs = []
            for class_name in classes:
                cls_objs = [obj for obj in objs if obj['class']==class_name]

                max_score = 0
                max_obj = None
                for obj in cls_objs:
                    if obj['class_score'] > max_score:
                        max_obj = obj
                        max_score = obj['class_score']

                new_objs.append(max_obj)

            return new_objs

    class gpt4_model():
        def __init__(self):
            print('Registering gpt4 model')

        def predict(self,prompt,max_tokens=128):
            import time
            wait_time = 5
            time.sleep(wait_time)
            response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0
                )
            return response


    @classmethod
    def init_models(cls):
        cls.__vqa_model = cls.vqa_model()
        cls.__loc_model = cls.loc_model()

        cls.__glip_model = cls.glip_model()
        cls.__maskrcnn_model = cls.maskrcnn_model()

        cls.__object_detector_model = cls.object_detector_model()
        cls.__depth_model = cls.depth_model()
        cls.__clip_model = cls.clip_model()
        cls.__gpt3_model = cls.gpt3_model()
        cls.__xvlm_model = cls.xvlm_model()

        cls.__face_detection_model = cls.face_detection_model()

        cls.__segment_model = cls.segment_model()
        cls.__select_model = cls.select_model()
        cls.__replace_model = cls.replace_model()
        cls.__query_obj_model = cls.query_obj_model()
    @classmethod
    def clip(cls, image: Image.Image, prompt: str) -> str:
        """
        Get response from clip
        Input:
            image: image to clip
            prompt: prompt to clip
        Output:
            response: clip response
        """
        # minor update
        return cls.__clip_model.predict(image,prompt=prompt,task='score')

    @classmethod
    def vqa(cls, image: Image.Image, question: str) -> str:
        """
        Answer simple question about the image.
        Input:
            image: an image object
            question: a text string
        Output:
            answer: a text string
        """
        return cls.__vqa_model.predict(image, question)

    @classmethod
    def loc(cls, image: Image.Image, object: str, thresh=0.1, nms_thresh=0.5) -> List[List[int]]:
        """
        Generate boxes of the object on the image.
        Input:
            image: an image object
            object: a text string
        Output:
            boxes: a list of bounding boxes
        """
        return cls.__loc_model.predict(image, object, thresh, nms_thresh)

    @classmethod
    def find(cls, image: Image.Image, object: str, glip_thresh=0.5) -> List[List[int]]:
        if object in ["object", "objects"]:
            all_object_coordinates = cls.__maskrcnn_model.predict(image)[0]
        else:

            if object == 'person':
                object = 'people'  # GLIP does better at people than person

            all_object_coordinates = cls.__glip_model.predict(image, object, confidence_threshold=glip_thresh)

        return all_object_coordinates


    @classmethod
    def object_detector(cls, image: Image.Image) -> List[List[int]]:
        """
        Get object detection boxes
        Input:
            image: an image object
        Output:
            boxes: a list of bounding boxes
        """
        return cls.__object_detector_model.predict(image)

    @classmethod
    def depth(cls, image: Image.Image) -> List[List[float]]:
        """
        Estimate depth map
        Input:
            image: an image object
        Output:
            map: a depth map
        """
        return cls.__depth_model.predict(image)

    @classmethod
    def glip(cls, image: Image.Image, object: str) -> List[List[int]]:
        """
        Generate boxes of the object on the image.
        Input:
            image: an image object
            object: a text string
        Output:
            boxes: a list of bounding boxes
        """
        return cls.__glip_model.predict(image, object)

    @classmethod
    def gpt3(cls, prompt: str, process_name='gpt3_general') -> str:
        """
        Get response from gpt3
        Input:
            prompt: prompt to gpt3
        Output:
            response: gpt3 response
        """
        return cls.__gpt3_model.predict(prompt, process_name)

    @classmethod
    def blip(cls, image: Image.Image) -> str:
        """
        Caption the image
        Input:
            image: an image object
        Output:
            caption: a text string
        """
        return cls.__blip_model.predict(image)

    @classmethod
    def xvlm(cls, image: Image.Image, category: str, negative_categories: List[str]) -> float:
        """
        Returns a binary score for the similarity between the image and the category.
        The negative categories are used to compare to (score is relative to the scores of the negative categories).
        Input:
            image: an image object
            category: a text string
            negative_categories: a list of text strings
        Output:
            score: a float number
        """
        return cls.__xvlm_model.predict(image, category, negative_categories).item()

    @classmethod
    def text_match(cls, image: Image.Image, texts: List[str]) -> str:
        """
        Returns the string that best matches the image.
        Input:
            image: an image object
            texts: a list of text strings
        Output:
            answer: the best matched text
        """
        res = cls.__xvlm_model.predict(image, texts, task='score')
        selected = res.argmax().item()
        return texts[selected]

    @classmethod
    def select_answer(cls, info: dict, question: str, options: List[str]) -> str:
        """
        Returns the answer to the question given the options and additional information.
        Input:
            info: a dict
            question: a text string
            options: a list of text strings
        Output:
            answer: the predicted answer
        """
        prompt = cls.select_answer_prompt
        prompt = prompt.replace('__INSERT_QUESTION__', question)
        option_str = ''
        for index, option in enumerate(options):
            option_str += f"({index}) {option} "
        prompt = prompt.replace('__INSERT_OPTIONS__', option_str)
        prompt = prompt.replace('__INSERT_INFO__', json.dumps(info))
        return cls.__gpt3_model.predict(prompt, process_name='gpt3_general')

    @classmethod
    def fix_caption(cls, caption: str, objects: List[str]) -> str:
        prompt = cls.fix_caption_prompt
        prompt = prompt.replace('__INSERT_CAPTION__', caption)
        prompt = prompt.replace('__INSERT_OBJECTS__', ','.join(objects))
        response = cls.__gpt3_model.predict(prompt, process_name='gpt3_general')
        return response.split('Modified caption:')[1].strip()

    @classmethod
    def vqa_long(cls, image: Image.Image, question: str) -> str:
        return cls.__blip_model.predict(image, question, 'qa_long')

    @classmethod
    def vqa_short(cls, image: Image.Image, question: str) -> str:
        return cls.__blip_model.predict(transforms.ToTensor()(image), question, 'qa')

    @classmethod
    def blip_vqa(cls, image: Image.Image, question: str) -> str:
        return cls.__blip_model.predict(image, question, 'qa_test')

    @classmethod
    def face_detection(cls, image: Image.Image):
        return cls.__face_detection_model.predict(image)

    @classmethod
    def segment(cls, image: Image.Image):
        return cls.__segment_model.predict(image)

    @classmethod
    def select(cls, query, objects, image):
        return cls.__select_model.query_obj(query, objects, image)

    @classmethod
    def replace(cls, prompt, image, mask_image, guidance_scale=7.5, num_inference_steps=50):
        return cls.__replace_model.predict(prompt, image, mask_image, guidance_scale, num_inference_steps)

    @classmethod
    def query_obj(cls, query, objs, img):
        return cls.__query_obj_model.predict(query, objs, img)

