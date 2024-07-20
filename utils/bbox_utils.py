def get_box_center(bbox):
    return (bbox[0] + bbox[2]) // 2
def get_box_width(bbox):
    return bbox[2]-bbox[0]