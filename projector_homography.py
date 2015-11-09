import numpy as np
import cv2
import cv

# this just handles actually showing the window 
# and the dots where you've clicked
class SelectView:
    def __init__(self, winname, imsize):
        self.im = np.zeros((imsize, imsize, 3), dtype=np.uint8)
        self.clicks = []
        self.winname = winname
        cv2.namedWindow(self.winname)
        cv.SetMouseCallback(self.winname, self.mouseHandler, 0)

    def addClick(self, x, y):
        self.clicks.append((x,y))

    def mouseHandler(self, event, x, y, flags, params):
        if event == cv.CV_EVENT_LBUTTONDOWN:
            self.addClick(x, y)

    def renderWindow(self):
        self.dispim = self.im.copy()
        for (x, y) in self.clicks:
            cv2.circle(self.dispim, (int(x), int(y)), 8, (255,255,255), 2)
        cv2.imshow(self.winname, self.dispim)

    def finishSelection(self):
        cv2.destroyWindow(self.winname)

# this handles the actual math for computing the homography
def compute_homography(srcpoints, destpoints):
    src_pts = np.array([ list(p) for p in srcpoints ], dtype=np.float32).reshape(1,-1,2)
    dst_pts = np.array([ list(p) for p in destpoints ], dtype=np.float32).reshape(1,-1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(M)
    return M

def compute_perspective(srcpoints, destpoints):
    src_pts = np.array([ list(p) for p in srcpoints ], dtype=np.float32).reshape(1,-1,2)
    dst_pts = np.array([ list(p) for p in destpoints ], dtype=np.float32).reshape(1,-1,2)

    return cv2.getPerspectiveTransform(src_pts, dst_pts)

def warp_image(srcim, H, invert=False):
    if invert:
        Hp = np.linalg.inv(H)
    else:
        Hp = H

    return cv2.warpPerspective(srcim, Hp, (srcim.shape[0], srcim.shape[1]))

if __name__ == '__main__':
    imsize = 1024

    # get correspondences through 'gui'
    clickview = SelectView("selectview", imsize)
    while True:
        clickview.renderWindow()
        if len(clickview.clicks) == 4:
            break
        keycode = cv.WaitKey(30)
    clickview.finishSelection()
    print(clickview.clicks)

    # compute perspective transform (you can save M to reuse later)
    destpoints = [(0,0), (imsize,0), (imsize,imsize), (0, imsize)]
    M = compute_perspective(clickview.clicks, destpoints)
    print(M)

    # warp image
    inimage = cv2.imread("test.png")
    warpimage = warp_image(inimage, M, True)
    cv2.imshow("warped", warpimage)
    cv.WaitKey(0)