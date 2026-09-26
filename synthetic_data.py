import numpy as np
import cv2
import random

def make_clear(h=64, w=64, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    img = np.zeros((h, w, 3), np.float32)
    for y in range(h):
        t = y / max(1, h - 1)
        img[y, :, 0] = .04 + .10 * t
        img[y, :, 1] = .06 + .12 * t
        img[y, :, 2] = .12 + .20 * t
    pts = np.array([[int(.32*w),h],[int(.68*w),h],[int(.56*w),int(.55*h)],[int(.44*w),int(.55*h)]], np.int32)
    cv2.fillConvexPoly(img, pts, (.11,.11,.12))
    cv2.line(img, (w//2,h), (w//2,int(.58*h)), (.7,.68,.55), 1)
    for x in [int(.15*w),int(.28*w),int(.72*w),int(.85*w)]:
        y = random.randint(int(.35*h), int(.65*h))
        cv2.line(img, (x,y), (x,h), (.08,.09,.10), 1)
        cv2.circle(img, (x,y), 2, (1,.75,.35), -1)
    return np.clip(img + np.random.normal(0,.008,img.shape), 0, 1).astype(np.float32)

def degrade(c, seed=None):
    if seed is not None: np.random.seed(seed)
    x = np.clip(c*.45 + np.array([.015,.02,.04]), 0, 1)
    h,w,_ = x.shape
    Y,X = np.mgrid[0:h,0:w]
    cc = np.random.uniform(.2,.8,2)
    d = (X/w-cc[0])**2 + (Y/h-cc[1])**2
    haze = np.clip(.15+.5*np.exp(-d/.25),0,.65)[...,None]
    x = x*(1-haze) + np.array([.55,.60,.68])*haze
    for _ in range(np.random.randint(1,4)):
        cx=np.random.randint(8,w-8); cy=np.random.randint(int(.35*h),int(.85*h))
        yy,xx=np.mgrid[0:h,0:w]
        s=np.random.uniform(2,5)
        g=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*s*s))[...,None]
        x=np.clip(x+g*np.array([1,.85,.45])*np.random.uniform(.35,.7),0,1)
    return np.clip(x+np.random.normal(0,.015,x.shape),0,1).astype(np.float32)
