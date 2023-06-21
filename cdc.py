import numpy as np
import scipy
import os
import cv2

# https://github.com/lyh-18/TCVC-Temporally-Consistent-Video-Colorization/blob/95bef8d1caf9a098112f107910995cedd61e7280/codes/eval_results.py#L142

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p,M)+0.5*scipy.stats.entropy(q, M)
    
def compute_JS_bgr(input_dir, dilation=1):
    input_img_list = os.listdir(input_dir)
    input_img_list.sort()
    #print(input_img_list)

    hist_b_list = []   # [img1_histb, img2_histb, ...]
    hist_g_list = []
    hist_r_list = []
    
    for img_name in input_img_list:
        #print(os.path.join(input_dir, img_name))
        img_in = cv2.imread(os.path.join(input_dir,  img_name))
        H, W, C = img_in.shape
        
        hist_b = cv2.calcHist([img_in],[0],None,[256],[0,256]) # B
        hist_g = cv2.calcHist([img_in],[1],None,[256],[0,256]) # G
        hist_r = cv2.calcHist([img_in],[2],None,[256],[0,256]) # R
        
        hist_b = hist_b/(H*W)
        hist_g = hist_g/(H*W)
        hist_r = hist_r/(H*W)
        
        hist_b_list.append(hist_b)
        hist_g_list.append(hist_g)
        hist_r_list.append(hist_r)
    
    JS_b_list = []
    JS_g_list = []
    JS_r_list = []

    for i in range(len(hist_b_list)):
        if i+dilation > len(hist_b_list)-1:
            break
        hist_b_img1 = hist_b_list[i]
        hist_b_img2 = hist_b_list[i+dilation]     
        JS_b = JS_divergence(hist_b_img1, hist_b_img2)
        JS_b_list.append(JS_b)
        
        hist_g_img1 = hist_g_list[i]
        hist_g_img2 = hist_g_list[i+dilation]     
        JS_g = JS_divergence(hist_g_img1, hist_g_img2)
        JS_g_list.append(JS_g)
        
        hist_r_img1 = hist_r_list[i]
        hist_r_img2 = hist_r_list[i+dilation]     
        JS_r = JS_divergence(hist_r_img1, hist_r_img2)
        JS_r_list.append(JS_r)
        
        '''
        plt.subplot(1,2,1)
        plt.imshow(img_in[:,:,[2,1,0]])
        plt.subplot(1,2,2)
        plt.plot(hist_b_img1)
        plt.plot(hist_b_img2)
        plt.show()
        '''

        CDC = np.mean([float(np.mean(JS_b_list)), float(np.mean(JS_g_list)), float(np.mean(JS_r_list))])
        
    return CDC

if __name__ == "__main__":
    input_folder = "C:/video_colorization/data/train/mini_DAVIS/drone/"
    # input_folder = r"C:\video_colorization\vit_colorization\temp_result\mini_DAVIS\swin_unet_20230620_100226\bus.mp4"
    input_folder = r"C:\video_colorization\vit_colorization\temp_result\DAVIS_test\swin_unet_20230619_110942\aerobatics.mp4"

    dilation = [1,2,4]
    weight = [1/3, 1/3, 1/3]
    # aa = calculate_folders(input_folder, input_folder, dilation=dilation)
    aa = compute_JS_bgr(input_folder, dilation=1)
    print(aa)
    # JS_b_mean_list_1, JS_g_mean_list_1, JS_r_mean_list_1, JS_b_dict_1, JS_g_dict_1, JS_r_dict_1, CDC = calculate_folders_multiple(input_folder, input_folder, dilation=dilation, weight=weight)

    # p = torch.randn((2, 3, 224, 224))
    # q = torch.randn((2, 3, 224, 224))

    # jsd = JS_divergence
    # aa = jsd(p, q)