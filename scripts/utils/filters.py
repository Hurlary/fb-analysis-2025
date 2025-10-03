
def looks_like_ball(box):
    x1, y1, x2, y2 = box.xyxy[0]
    w, h = x2 - x1, y2 - y1
    aspect_ratio = w / h if h != 0 else 0
    area = w * h
    return 0.75 < aspect_ratio < 1.25 and 100 < area < 1000
