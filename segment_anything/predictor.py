# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from segment_anything.modeling import Sam

from typing import Optional, Tuple

from .utils.transforms import ResizeLongestSide

from typing import Union
import torch.nn.functional as F
# from transformers import CLIPTokenizer
from typing import List, Tuple
from typing import Optional, List
import cv2

# # 初始化 tokenizer
# tokenizer = CLIPTokenizer.from_pretrained("/home/chenlin/Desktop/segment-anything/openai/clip-vit-base-path16")



class SamPredictor:
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def transform_points(
            self,
            points, 
            H_orig, 
            W_orig, 
            long_side_length=1024, 
            final_size=64
    ):
        """
        变换 point 坐标，使其匹配 64×64 特征图，严格按照 `F.pad` 逻辑：
        - 先缩放，使最长边等于 `1024`
        - 如果 `H < W`，则下方填充
        - 如果 `H > W`，则右侧填充
        - 最后映射到 `64×64`
        """
        # Step 1: 计算缩放比例
        print("H_orig:",H_orig)
        print("W_orig:",W_orig)
        # 确保 H_orig 和 W_orig 是 Python 标量
        if isinstance(H_orig, torch.Tensor):
            H_orig = H_orig.item()  # 取出单个数值
        if isinstance(W_orig, torch.Tensor):
            W_orig = W_orig.item()
        scale = long_side_length / max(H_orig, W_orig)
        points_scaled = [(x * scale, y * scale) for x, y in points]

        # Step 2: 进一步缩放到 1024×1024
        max_side = max(int(H_orig * scale), int(W_orig * scale))
        if max_side != 1024:
            scale_factor = 1024 / max_side
            points_scaled = [(x * scale_factor, y * scale_factor) for x, y in points_scaled]

        # Step 4: 映射到 `64×64`
        # item将张量转换为标量 round进行四舍五入
        points_final = [(round(x.item() / 16), round(y.item() / 16)) for x, y in points_scaled]

        # 确保坐标值不超过 63
        points_final = [(min(x, 63), min(y, 63)) for x, y in points_final]
        points_final = torch.tensor(points_final)

        return points_final

    def transform_box(
        self,
        box,  # 输入必须是张量
        H_orig: int,
        W_orig: int,
        long_side_length: int = 1024,
        final_size: int = 64
    ) -> Optional[torch.Tensor]:
        box = torch.tensor(box)
        if box is None:
            return None

        # 确保输入张量形状为 [N=1,4] 或 [4]
        box = box.view(-1)  # 展平为 [4]
        if box.numel() != 4:
            return None  # 输入不合法

        # Step 1: 转换为浮点数
        box = box.to(torch.int32)

        x_min, y_min, x_max, y_max = box

        # Step 2: 计算缩放比例
        scale = long_side_length / max(H_orig, W_orig)

        # Step 3: 缩放坐标
        x_min_scaled = x_min * scale
        y_min_scaled = y_min * scale
        x_max_scaled = x_max * scale
        y_max_scaled = y_max * scale

        # Step 4: 映射到64x64网格
        x1 = torch.round(x_min_scaled / 16).int()
        y1 = torch.round(y_min_scaled / 16).int()
        x2 = torch.round(x_max_scaled / 16).int()
        y2 = torch.round(y_max_scaled / 16).int()

        # Step 5: 约束范围
        x1 = torch.clamp(x1, 0, 63)
        y1 = torch.clamp(y1, 0, 63)
        x2 = torch.clamp(x2, 0, 63)
        y2 = torch.clamp(y2, 0, 63)

        box_final = torch.tensor([x1, y1, x2, y2], dtype=torch.int32, device=box.device)

        # 返回张量（形状 [4], dtype=int32）
        return box_final
    # def set_image(
    #     self,
    #     points: List[Tuple[int, int]],
    #     image: np.ndarray,
    #     # feature_map: torch.Tensor,  # 新增 feature_map 参数
    #     image_format: str = "RGB",
    # ) -> None:
    #     """
    #     Calculates the image embeddings for the provided image, allowing
    #     masks to be predicted with the 'predict' method.

    #     Arguments:
    #       image (np.ndarray): The image for calculating masks. Expects an
    #         image in HWC uint8 format, with pixel values in [0, 255].
    #       image_format (str): The color format of the image, in ['RGB', 'BGR'].
    #     """
    #     assert image_format in [
    #         "RGB",
    #         "BGR",
    #     ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
    #     if image_format != self.model.image_format:
    #         image = image[..., ::-1]

    #     # Transform the image to the form expected by the model
    #     input_image = self.transform.apply_image(image)
    #     input_points = self.transform_points(points, image.shape[0], image.shape[1])
    #     input_image_torch = torch.as_tensor(input_image, device=self.device)
    #     input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    #     self.set_torch_image(input_image_torch, image.shape[:2], input_points)

    def set_image(
            self,
            image: np.ndarray,# 非默认参数必须在默认参数之前
            points: Optional[List[Tuple[int, int]]] = None,
            box: Optional[List[int]] = None,
            feature_map: torch.Tensor = None,  # 新增 feature_map 参数
            image_format: str = "RGB",
        ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          points (Optional[List[Tuple[int, int]]]): A list of point prompts to the
            model. Each point is in (X, Y) in pixels. Default is None.
          box_coords (Optional[torch.Tensor]): A tensor of shape (1, 4) representing
            a bounding box prompt to the model, in XYXY format. Default is None.
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        # Transform points if provided
        if points is not None:
            input_points = self.transform_points(points, image.shape[0], image.shape[1])
            input_points = input_points.to(self.device)
        else:
            input_points = None

        # Ensure box_coords is on the correct device if provided
        if box is not None:
            # print("box.shape", box.shape)
            input_box = self.transform_box(box, image.shape[0], image.shape[1])
            # print("input_box:", input_box)
            input_box = input_box.to(self.device)
        else:
            input_box = None

        loss_align = self.set_torch_image(input_image_torch, image.shape[:2], input_points, input_box, feature_map)
        # self.set_torch_image(input_image_torch, image.shape[:2], input_points, input_box, feature_map)
        
        return loss_align
        
    # def set_image(
    #     self,
    #     image: Union[np.ndarray, torch.Tensor],  # 允许 PyTorch 张量和 NumPy 数组
    #     image_format: str = "RGB",
    # ) -> None:
    #     """
    #     Calculates the image embeddings for the provided image, allowing
    #     masks to be predicted with the 'predict' method.

    #     Arguments:
    #       image (np.ndarray or torch.Tensor): The image for calculating masks. Expects an
    #         image in HWC format, with pixel values in [0, 255].
    #       image_format (str): The color format of the image, in ['RGB', 'BGR'].
    #     """
    #     assert image_format in [
    #         "RGB",
    #         "BGR",
    #     ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."

    #     if isinstance(image, torch.Tensor):
    #         # 直接使用 GPU 张量，不需要转换回 NumPy
    #         input_image_torch = image.permute(2, 0, 1).contiguous()[None, :, :, :]
    #     else:
    #         if image_format != self.model.image_format:
    #             image = image[..., ::-1]

    #         input_image = self.transform.apply_image(image)
    #         input_image_torch = torch.as_tensor(input_image, device=self.device)
    #         input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    #     print(f"input_image_torch.shape: {input_image_torch.shape}")  # 打印张量形状
    #     print(f"Expected BCHW format with max size 1024")

    #     self.set_torch_image(input_image_torch, image.shape[:2])


    # @torch.no_grad()
    # def set_torch_image(
    #     self,
    #     transformed_image: torch.Tensor,
    #     original_image_size: Tuple[int, ...],
    # ) -> None:
    #     """
    #     Calculates the image embeddings for the provided image, allowing
    #     masks to be predicted with the 'predict' method. Expects the input
    #     image to be already transformed to the format expected by the model.

    #     Arguments:
    #       transformed_image (torch.Tensor): The input image, with shape
    #         1x3xHxW, which has been transformed with ResizeLongestSide.
    #       original_image_size (tuple(int, int)): The size of the image
    #         before transformation, in (H, W) format.
    #     """
    #     assert (
    #         len(transformed_image.shape) == 4
    #         and transformed_image.shape[1] == 3
    #         and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
    #     ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
    #     self.reset_image()

    #     self.original_size = original_image_size
    #     self.input_size = tuple(transformed_image.shape[-2:])
    #     input_image = self.model.preprocess(transformed_image)
    #     self.features = self.model.image_encoder(input_image)
    #     self.is_image_set = True
    # @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
        input_points: torch.Tensor,  # 新增参数
        input_box: torch.Tensor,  # 新增参数
        feature_map: torch.Tensor = None,  # 新增 feature_map 参数
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        # Debugging: print the shape of the input image
        print(f"Original transformed_image shape: {transformed_image.shape}")

        # If the image is [H, W, 3] or [H, W, C] we need to convert to [B, C, H, W]
        if len(transformed_image.shape) == 4 and transformed_image.shape[-1] == 3:
            # Convert [1, H, W, 3] to [1, 3, H, W] by permuting the dimensions
            transformed_image = transformed_image.permute(0, 3, 1, 2)
        elif len(transformed_image.shape) == 3 and transformed_image.shape[2] == 3:
            # Convert [H, W, 3] to [1, 3, H, W]
            transformed_image = transformed_image.permute(2, 0, 1).unsqueeze(0)
        elif len(transformed_image.shape) != 4 or transformed_image.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W] or [H, W, 3] format, but got {transformed_image.shape}")

        # Print the shape again after transformation
        print(f"Transformed transformed_image shape: {transformed_image.shape}")

        # Ensure that the image's channel dimension is 3
        assert transformed_image.shape[1] == 3, f"Expected 3 channels, but got {transformed_image.shape[1]} channels"

        # Adjust the size if the longest side is not 1024
        H, W = transformed_image.shape[2], transformed_image.shape[3]
        max_side = max(H, W)
        if max_side != 1024:
            scale_factor = 1024 / max_side
            new_H = int(H * scale_factor)
            new_W = int(W * scale_factor)
            transformed_image = F.interpolate(
                transformed_image, size=(new_H, new_W), mode='bilinear', align_corners=False
            )

        # Now the transformed image has the correct size
        assert (
            max(*transformed_image.shape[2:]) == 1024
        ), f"set_torch_image input must be BCHW with long side 1024, current shape: {transformed_image.shape}"

        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        # print("input_image.max:", torch.max(input_image))
        # print("input_image.min:", torch.min(input_image))
        # print("input_image.device:", input_image.device)
        # print("input_image.shape:", input_image.shape)
        # # 1. 将张量从 GPU 移到 CPU 并移除 batch 维度
        # image = input_image.squeeze(0)  # 从 [1, 3, 1024, 1024] 到 [3, 1024, 1024]
        # image = image.cpu()  # 移到 CPU

        # # 2. 将张量转换为 NumPy 数组并调整维度
        # image = image.permute(1, 2, 0)  # 从 [3, 1024, 1024] 到 [1024, 1024, 3]
        # image = image.numpy()  # 转换为 NumPy 数组

        # # 3 ARRAY IS NOW IN RGB, convert to BGR for OpenCV
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # # 4. 确保像素值在 [0, 255] 范围内（如果张量是浮点数）
        # if image.dtype != np.uint8:
        #     # 假设张量值在 [0, 1] 范围内（常见于归一化后的图像）
        #     image = (image * 255).astype(np.uint8)

        # # 5. 保存图像
        # cv2.imwrite('results/Display/samples/12/reshape/1.png', image)
        # exit()
        # if feature_map!=None:
        #   print("feature_map.shape:", feature_map.shape)
        # else:
        #   print("None in featureMap!")
        self.features, loss_align= self.model.image_encoder(input_image, input_points, input_box, feature_map)
        self.is_image_set = True
        
        return loss_align

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].detach().cpu().numpy()
        # masks_np = masks.detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        # low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks[0]
        # return masks, iou_predictions, low_res_masks
    
    # @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            # Ensure point_coords is a BxNx2 tensor and point_labels is BxN tensor
            if point_coords.ndimension() == 2:
                point_coords = point_coords.unsqueeze(0)  # Adding batch dimension (B=1)
            
            assert point_coords.ndimension() == 3 and point_coords.shape[2] == 2, \
                f"Expected point_coords shape BxNx2, but got {point_coords.shape}"
            
            if point_labels.ndimension() == 1:
                # If point_labels is 1D, add a batch dimension (B=1) and make it 2D
                point_labels = point_labels.unsqueeze(0)  # Adding batch dimension (B=1)
            
            assert point_labels.ndimension() == 2 and point_labels.shape[1] == point_coords.shape[1], \
                f"Expected point_labels shape BxN, but got {point_labels.shape}"
            
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        
        print("sparse_embeddings.shape after prompt_encoder:", sparse_embeddings.shape)

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        print("low_res_masks.shape:", low_res_masks.shape)
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        print("masks.shape in predict_torch", masks.shape)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks
    
    # @torch.no_grad()
    def predict_torch_promptEncoder(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """

        if point_coords is not None:
            # Ensure point_coords is a BxNx2 tensor and point_labels is BxN tensor
            if point_coords.ndimension() == 2:
                point_coords = point_coords.unsqueeze(0)  # Adding batch dimension (B=1)
            
            assert point_coords.ndimension() == 3 and point_coords.shape[2] == 2, \
                f"Expected point_coords shape BxNx2, but got {point_coords.shape}"
            
            if point_labels.ndimension() == 1:
                # If point_labels is 1D, add a batch dimension (B=1) and make it 2D
                point_labels = point_labels.unsqueeze(0)  # Adding batch dimension (B=1)
            
            assert point_labels.ndimension() == 2 and point_labels.shape[1] == point_coords.shape[1], \
                f"Expected point_labels shape BxN, but got {point_labels.shape}"
            
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings, feature_map = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        
        print("sparse_embeddings.shape after prompt_encoder:", sparse_embeddings.shape)

        return sparse_embeddings, dense_embeddings, feature_map
  
    # @torch.no_grad()
    def predict_torch_maskDecoder(
        self,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    # @torch.no_grad()
    # def predict_torch(
    #     self,
    #     point_coords: Optional[torch.Tensor],
    #     point_labels: Optional[torch.Tensor],
    #     boxes: Optional[torch.Tensor] = None,
    #     mask_input: Optional[torch.Tensor] = None,
    #     text: Optional[str] = None,  # 传入文本
    #     multimask_output: bool = True,
    #     return_logits: bool = False,
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     if not self.is_image_set:
    #         raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

    #     # 处理点提示
    #     if point_coords is not None:
    #         if point_coords.ndimension() == 2:
    #             point_coords = point_coords.unsqueeze(0)  # 添加 batch 维度
    #         assert point_coords.ndimension() == 3 and point_coords.shape[2] == 2, \
    #             f"Expected point_coords shape BxNx2, but got {point_coords.shape}"
            
    #         if point_labels.ndimension() == 1:
    #             point_labels = point_labels.unsqueeze(0)  # 添加 batch 维度
    #         assert point_labels.ndimension() == 2 and point_labels.shape[1] == point_coords.shape[1], \
    #             f"Expected point_labels shape BxN, but got {point_labels.shape}"
            
    #         points = (point_coords, point_labels)
    #     else:
    #         points = None

    #     # 处理文本提示
    #     tokens = None
    #     if text is not None:
    #         tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids
    #         tokens = tokens.to(self.device)  # 确保 tokens 在正确的设备上

    #     # 编码点、框、掩码、文本提示
    #     sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
    #         points=points,
    #         boxes=boxes,
    #         masks=mask_input,
    #         tokens=tokens  # 传入 tokenized 文本
    #     )

    #     # 预测掩码
    #     low_res_masks, iou_predictions = self.model.mask_decoder(
    #         image_embeddings=self.features,
    #         image_pe=self.model.prompt_encoder.get_dense_pe(),
    #         sparse_prompt_embeddings=sparse_embeddings,
    #         dense_prompt_embeddings=dense_embeddings,
    #         multimask_output=multimask_output,
    #     )

    #     # 还原到原始尺寸
    #     masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

    #     if not return_logits:
    #         masks = masks > self.model.mask_threshold

    #     return masks, iou_predictions, low_res_masks




    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
