
########################metrics.py#########其中FocalerLoss在这里面实现---参考Snu77的实现#########
def bbox_iou(box1, box2, epoch=1,xywh=True, GIoU=False, DIoU=False, CIoU=False,ShapeIoU=False,UIoU=False,Focaler=False,Inner=False,PIoU=False, ratio=0.7,eps=1e-7,scale=0.0):
    if UIoU:
            epoch = epoch + 1
            ################缩放尺度的中心点坐标######### # define the center point for scaling#######
            bb1_xc = x1
            bb1_yc = y1
            bb2_xc = x2
            bb2_yc = y2
            # attenuation mode of hyperparameter "ratio"#############衰减模型的ratio
            linear = True
            cosine = False
            fraction = False
            if linear:
                ratio = -0.005 * epoch + 2  #################自己控制
            elif cosine:
                ratio = 0.75 * math.cos(math.pi * epoch / 300) + 1.25 ################自己控制  默认可以选择这个
            elif fraction:
                ratio = 200 / (epoch + 100)
            else:
                ratio = 0.5
            ww1, hh1, ww2, hh2 = w1 * ratio, h1 * ratio, w2 * ratio, h2 * ratio################根据ratio计算调整后的宽度和高度
            bb1_x1, bb1_x2, bb1_y1, bb1_y2 = bb1_xc - (ww1 / 2), bb1_xc + (ww1 / 2), bb1_yc - (hh1 / 2), bb1_yc + (hh1 / 2) ####计算调整后的左下角右下角坐标
            bb2_x1, bb2_x2, bb2_y1, bb2_y2 = bb2_xc - (ww2 / 2), bb2_xc + (ww2 / 2), bb2_yc - (hh2 / 2), bb2_yc + (hh2 / 2)############
            # assign the value back to facilitate subsequent calls   调整后的值返回赋值
            w1, h1, w2, h2 = ww1, hh1, ww2, hh2
            b1_x1, b1_x2, b1_y1, b1_y2 = bb1_x1, bb1_x2, bb1_y1, bb1_y2
            b2_x1, b2_x2, b2_y1, b2_y2 = bb2_x1, bb2_x2, bb2_y1, bb2_y2
    
    if Inner:    # transform from xywh to xyxy
        if not xywh:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
            #box1, box2 = box1.chunk(4, -1), box2.chunk(4, -1)
        #(x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)    ############box2为真实框
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - (w1 * ratio) / 2, x1 + (w1 * ratio) / 2, y1 - (h1 * ratio) / 2, y1 + (h1 * ratio) / 2
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - (w2 * ratio) / 2, x2 + (w2 * ratio) / 2, y2 - (h2 * ratio) / 2, y2 + (h2 * ratio) / 2
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
 
        union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
        '''
        inter_width= (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0)
        inter_height= (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
        inter_x1= b1_x1.minimum(b2_x1)
        inter_x2= b1_x2.maximum(b2_x2)
        inter_center_x= (inter_x1 + inter_x2)/2
        inter_center_y= (inter_y1 + inter_y2)/2
        '''
    else:
        # Get the coordinates of bounding boxes
        if xywh:  # transform from xywh to xyxy
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        ).clamp_(0)

        # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
     
    iou=inter / union ####Inner_iou 
    if Focaler:
        
        iou = ((iou - d) / (u - d)).clamp(0, 1)  # default d=0.00,u=0.95 

    if CIoU or DIoU or GIoU or PIoU or ShapeIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or OIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            
            return iou - rho2 / c2  # DIoU
        elif ShapeIoU:
                #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance
                ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
                ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
                c2 = cw ** 2 + ch ** 2 + eps                            # convex diagonal squared
                center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
                center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                center_distance = hh * center_distance_x + ww * center_distance_y
                distance = center_distance / c2
 
                #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape
                omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - distance - 0.5 * shape_cost
        elif PIoU:
            dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))
            dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))
            dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))
            dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))
            P = ((dw1 + dw2) / torch.abs(w2) + (dh1 + dh2) / torch.abs(h2)) / 4
            L_v1 = 1 - iou - torch.exp(-P ** 2) + 1
            iou =L_v1
            if PIoU2:
                q = torch.exp(-P)
                x = q * Lambda
                iou= 3 * x * torch.exp(-x ** 2) * L_v1
            return iou
        
        else: ##################
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

####################################################metrics.py############################################################

###############################Loss.py##########################################################################
def  forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,epoch):
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], epoch,xywh=False,Focaler=True, Inner=True,PIoU=True)  ### Inner=True,PIoU=True为InnerPIoU
###Focaler=True, Inner=True,PIoU=True,ShapeIoU=True
###############################Loss.py##########################################################################
class v8DetectionLoss:类中
 def __call__(self, preds, batch):变成： def __call__(self, preds, batch,epoch):
在 # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
变成：
if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,epoch
            )
################################Loss.py##########################################################################
在task.py修改BaseModel的loss函数，增加epoch参数，设置默认为0，然后将其传入到criterion中