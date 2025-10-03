
def looks_like_ball(box):
    x1, y1, x2, y2 = box.xyxy[0]
    w, h = x2 - x1, y2 - y1
    aspect_ratio = w / h if h != 0 else 0
    area = w * h

    return 0.75 < aspect_ratio < 1.25 and 100 < area < 1000
def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox;
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]- bbox[0];