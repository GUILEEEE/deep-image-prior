import torch
import torch.nn as nn
from math import sqrt
import torch.nn.init
import kornia
from .common import *
from models import *
from models.regularizers import *
from models.skip import *

from utils.denoising_utils import *
#from utils.blur_utils import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


class TotalVariation3rdorderDeblurring(nn.Module):
	def __init__(self,input_depth, pad, img, diff_img_x, diff_img_y,diff_img_xx,diff_img_xy,diff_img_yy, diff_img_xxx, diff_img_xxy, diff_img_xyy, diff_img_yyy, beta, epsilon, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(TotalVariation3rdorderDeblurring,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon

		self.img0 = img

		self.diff_img0_x = diff_img_x
		self.diff_img0_y = diff_img_y

		self.diff_img0_xx = diff_img_xx
		self.diff_img0_xy = diff_img_xy
		self.diff_img0_yy = diff_img_yy

		self.diff_img0_xxx = diff_img_xxx
		self.diff_img0_xxy = diff_img_xxy
		self.diff_img0_xyy = diff_img_xyy
		self.diff_img0_yyy = diff_img_yyy

	def forward(self, input):	

		output = self.net(input)

		img = torch.squeeze(output) #[HxW]

		diff_img_x = derivative_central_x_greylevel(img,dtype)     #[HxW]
		diff_img_y = derivative_central_y_greylevel(img,dtype)     #[HxW]


		diff_img_xx = derivative_xx_greylevel(img,dtype) #[HxW]
		diff_img_xy = derivative_xy_greylevel(img,dtype) #[HxW]
		diff_img_yy = derivative_yy_greylevel(img,dtype) #[HxW]

		diff_img_xxx = derivative_xxx_greylevel(img,dtype) #[HxW]
		diff_img_xxy = derivative_xxy_greylevel(img,dtype) #[HxW]
		diff_img_xyy = derivative_xyy_greylevel(img,dtype) #[HxW]
		diff_img_yyy = derivative_yyy_greylevel(img,dtype) #[HxW]

		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		covariant_derivative_img_x = diff_img_x
		covariant_derivative_img_y = diff_img_y

		covariant_derivative_img_xx = diff_img_xx 
		covariant_derivative_img_xy = diff_img_xy 
		covariant_derivative_img_yy = diff_img_yy 
	
		covariant_derivative_img_xxx = diff_img_xxx
		covariant_derivative_img_xxy = diff_img_xxy
		covariant_derivative_img_xyy = diff_img_xyy 
		covariant_derivative_img_yyy = diff_img_yyy 
		
		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) 
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)
		g12 = torch.zeros((self.height,self.width)).type(torch.cuda.FloatTensor) #[HxW]
		detg = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) #[HxW]


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_img_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_img_x,covariant_derivative_img_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_img_y))


		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_img_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_img_xx,covariant_derivative_img_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_img_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_img_yy,covariant_derivative_img_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_img_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_img_xx,covariant_derivative_img_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_img_xy,covariant_derivative_img_xy))
						

		norm_regularizer3_squared = torch.mul(torch.pow(invg11,3),torch.square(covariant_derivative_img_xxx)) \
									+ torch.mul(torch.pow(invg22,3),torch.square(covariant_derivative_img_yyy)) \
									+2.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_yyy))\
									+6.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_img_xxy,covariant_derivative_img_xyy)) \
									+12.*torch.mul(torch.mul(torch.mul(invg11,invg22),invg12),torch.mul(covariant_derivative_img_xxy,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg11),invg12),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg22),invg12),torch.mul(covariant_derivative_img_yyy,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.square(covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.mul(covariant_derivative_img_yyy,covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.square(covariant_derivative_img_xyy)) \
									+3.*torch.mul(torch.mul(torch.square(invg11),invg22),torch.square(covariant_derivative_img_xxy)) \
									+3.*torch.mul(torch.mul(torch.square(invg22),invg11),torch.square(covariant_derivative_img_xyy))

		norm_regularizer1 = torch.sqrt(epsilon + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]

		norm_regularizer2 = torch.sqrt(epsilon + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]

		norm_regularizer3 = torch.sqrt(epsilon + norm_regularizer3_squared) #[HxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		

		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 

class RiemannianTotalVariation3rdorderDeblurring(nn.Module):
	def __init__(self,input_depth, pad, beta, epsilon, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(RiemannianTotalVariation3rdorderDeblurring,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon

	def forward(self, input):	

		output = self.net(input)

		img = torch.squeeze(output) #[HxW]

		diff_img_x = derivative_central_x_greylevel(img,dtype)     #[HxW]
		diff_img_y = derivative_central_y_greylevel(img,dtype)     #[HxW]


		diff_img_xx = derivative_xx_greylevel(img,dtype) #[HxW]
		diff_img_xy = derivative_xy_greylevel(img,dtype) #[HxW]
		diff_img_yy = derivative_yy_greylevel(img,dtype) #[HxW]

		diff_img_xxx = derivative_xxx_greylevel(img,dtype) #[HxW]
		diff_img_xxy = derivative_xxy_greylevel(img,dtype) #[HxW]
		diff_img_xyy = derivative_xyy_greylevel(img,dtype) #[HxW]
		diff_img_yyy = derivative_yyy_greylevel(img,dtype) #[HxW]

		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		covariant_derivative_img_x = diff_img_x
		covariant_derivative_img_y = diff_img_y

		covariant_derivative_img_xx = diff_img_xx 
		covariant_derivative_img_xy = diff_img_xy 
		covariant_derivative_img_yy = diff_img_yy 
	
		covariant_derivative_img_xxx = diff_img_xxx
		covariant_derivative_img_xxy = diff_img_xxy
		covariant_derivative_img_xyy = diff_img_xyy 
		covariant_derivative_img_yyy = diff_img_yyy 
		
		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_img_x)) #[HxW]
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_img_y)) #[HxW]
		g12 = self.beta*(torch.mul(covariant_derivative_img_x,covariant_derivative_img_y)) #[HxW]
		detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[HxW]


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_img_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_img_x,covariant_derivative_img_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_img_y))


		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_img_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_img_xx,covariant_derivative_img_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_img_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_img_yy,covariant_derivative_img_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_img_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_img_xx,covariant_derivative_img_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_img_xy,covariant_derivative_img_xy))
						

		norm_regularizer3_squared = torch.mul(torch.pow(invg11,3),torch.square(covariant_derivative_img_xxx)) \
									+ torch.mul(torch.pow(invg22,3),torch.square(covariant_derivative_img_yyy)) \
									+2.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_yyy))\
									+6.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_img_xxy,covariant_derivative_img_xyy)) \
									+12.*torch.mul(torch.mul(torch.mul(invg11,invg22),invg12),torch.mul(covariant_derivative_img_xxy,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg11),invg12),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg22),invg12),torch.mul(covariant_derivative_img_yyy,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.square(covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.mul(covariant_derivative_img_yyy,covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.square(covariant_derivative_img_xyy)) \
									+3.*torch.mul(torch.mul(torch.square(invg11),invg22),torch.square(covariant_derivative_img_xxy)) \
									+3.*torch.mul(torch.mul(torch.square(invg22),invg11),torch.square(covariant_derivative_img_xyy))

		norm_regularizer1 = torch.sqrt(epsilon + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]

		norm_regularizer2 = torch.sqrt(epsilon + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]

		norm_regularizer3 = torch.sqrt(epsilon + norm_regularizer3_squared) #[HxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		

		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 



class RiemannianTotalVariation3rdorderDeblurring2(nn.Module):
	def __init__(self,input_depth, pad, beta, epsilon, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(RiemannianTotalVariation3rdorderDeblurring2,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon

	def forward(self, input):	

		output = self.net(input)

		img = torch.squeeze(output) #[HxW]

		M = img.size(dim=0)
		N = img.size(dim=1)

		Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11 = Levi_Cevita_connection_coefficients_greylevel_standard_frame(img,self.beta,dtype)

		g11,g12,g22 = riemannianmetric_greylevel(img,self.beta,dtype)

		detg = torch.mul(g11,g22) - torch.mul(g12,g12) 
		invdetg = torch.div(torch.ones(M,N).type(dtype),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(dtype)


		norm_regularizer1 = norm_first_order_covariant_derivative_greylevel(img,epsilon,self.beta,dtype,invg11,invg12,invg22)							
		norm_regularizer2 = norm_second_order_covariant_derivative_greylevel(img,epsilon,self.beta,dtype,invg11,invg12,invg22,Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11)
		norm_regularizer3 = norm_third_order_covariant_derivative_greylevel(img,epsilon,self.beta,dtype,invg11,invg12,invg22,Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11)
						
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		
		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 


class RiemannianVectorialTotalVariation3rdorderDeblurring2(nn.Module):
	def __init__(self,input_depth, pad, beta, epsilon, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(RiemannianVectorialTotalVariation3rdorderDeblurring2,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon

	def forward(self, input):	

		output = self.net(input)

		img = torch.squeeze(output) #[3xHxW]

		M = img.size(dim=1)
		N = img.size(dim=2)

		Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11 = Levi_Cevita_connection_coefficients_color_standard_frame(img,self.beta,dtype)

		g11,g12,g22 = riemannianmetric_color(img,self.beta,dtype)

		detg = torch.mul(g11,g22) - torch.mul(g12,g12) 
		invdetg = torch.div(torch.ones(M,N).type(dtype),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)

		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(dtype)

		norm_regularizer1 = norm_first_order_covariant_derivative_color(img,epsilon,self.beta,dtype,invg11,invg12,invg22)							
		norm_regularizer2 = norm_second_order_covariant_derivative_color(img,epsilon,self.beta,dtype,invg11,invg12,invg22,Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11)
		norm_regularizer3 = norm_third_order_covariant_derivative_color(img,epsilon,self.beta,dtype,invg11,invg12,invg22,Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11)
						
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		
		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 




class VectorialTotalVariationRplusRplusRplus2ndorderDeblurring(nn.Module):
	def __init__(self,input_depth, pad, epsilon, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorialTotalVariationRplusRplusRplus2ndorderDeblurring,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.epsilon = epsilon

	def forward(self, input):	

		output = self.net(input)

		red = torch.narrow(output,1,0,1) #[1x1xHxW] 
		red = torch.squeeze(red) #[HxW]

		green = torch.narrow(output,1,1,1) #[1x1xHxW] 
		green = torch.squeeze(green) #[HxW]

		blue = torch.narrow(output,1,2,1) #[1x1xHxW] 
		blue = torch.squeeze(blue) #[HxW]

		diff_red_x = derivative_central_x_greylevel(red,dtype)     #[HxW]
		diff_green_x = derivative_central_x_greylevel(green,dtype) #[HxW]
		diff_blue_x = derivative_central_x_greylevel(blue,dtype)   #[HxW]

		diff_red_y = derivative_central_y_greylevel(red,dtype)     #[HxW]
		diff_green_y = derivative_central_y_greylevel(green,dtype) #[HxW]
		diff_blue_y = derivative_central_y_greylevel(blue,dtype)   #[HxW]

		diff_red_xx = derivative_xx_greylevel(red,dtype) #[HxW]
		diff_red_xy = derivative_xy_greylevel(red,dtype) #[HxW]
		diff_red_yy = derivative_yy_greylevel(red,dtype) #[HxW]

		diff_green_xx = derivative_xx_greylevel(green,dtype) #[HxW]
		diff_green_xy = derivative_xy_greylevel(green,dtype) #[HxW]
		diff_green_yy = derivative_yy_greylevel(green,dtype) #[HxW]

		diff_blue_xx = derivative_xx_greylevel(blue,dtype) #[HxW]
		diff_blue_xy = derivative_xy_greylevel(blue,dtype) #[HxW]
		diff_blue_yy = derivative_yy_greylevel(blue,dtype) #[HxW]

		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		covariant_derivative_red_x = diff_red_x
		covariant_derivative_red_y = diff_red_y

		covariant_derivative_green_x = diff_green_x
		covariant_derivative_green_y = diff_green_y

		covariant_derivative_blue_x = diff_blue_x
		covariant_derivative_blue_y = diff_blue_y

		covariant_derivative_red_xx = diff_red_xx
		covariant_derivative_red_xy = diff_red_xy
		covariant_derivative_red_yy = diff_red_yy
					
		covariant_derivative_green_xx = diff_green_xx
		covariant_derivative_green_xy = diff_green_xy
		covariant_derivative_green_yy = diff_green_yy
		
		covariant_derivative_blue_xx = diff_blue_xx
		covariant_derivative_blue_xy = diff_blue_xy
		covariant_derivative_blue_yy = diff_blue_yy

		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)
		g12 = torch.zeros((self.height,self.width)).type(torch.cuda.FloatTensor)
		detg = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_red_x)+torch.square(covariant_derivative_green_x)+torch.square(covariant_derivative_blue_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_red_x,covariant_derivative_red_y)+torch.mul(covariant_derivative_green_x,covariant_derivative_green_y)+torch.mul(covariant_derivative_blue_x,covariant_derivative_blue_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_red_y)+torch.square(covariant_derivative_green_y)+torch.square(covariant_derivative_blue_y))

		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_red_xx)+torch.square(covariant_derivative_green_xx)+torch.square(covariant_derivative_blue_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_red_xy)+torch.square(covariant_derivative_green_xy)+torch.square(covariant_derivative_blue_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_red_yy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_yy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_yy,covariant_derivative_blue_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_red_yy)+torch.square(covariant_derivative_green_yy)+torch.square(covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_yy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_yy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xy,covariant_derivative_blue_xy))
		

		norm_regularizer1 = torch.sqrt(epsilon + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]
	
		norm_regularizer2 = torch.sqrt(epsilon + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]


		return output, norm_regularizer1, norm_regularizer2	


class VectorBundleTotalVariationRplusRplusRplus2ndorderDeblurring(nn.Module):
	def __init__(self,input_depth, pad, red, green, blue, diff_red_x,diff_green_x,diff_blue_x, diff_red_y,diff_green_y,diff_blue_y,diff_red_xx,diff_red_xy,diff_red_yy,diff_green_xx,diff_green_xy,diff_green_yy,diff_blue_xx,diff_blue_xy,diff_blue_yy, beta, epsilon, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorBundleTotalVariationRplusRplusRplus2ndorderDeblurring,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon

		self.red0 = red
		self.green0 = green
		self.blue0 = blue

		self.diff_red0_x = diff_red_x
		self.diff_green0_x = diff_green_x
		self.diff_blue0_x = diff_blue_x
		self.diff_red0_y = diff_red_y
		self.diff_green0_y = diff_green_y
		self.diff_blue0_y = diff_blue_y

		self.diff_red0_xx = diff_red_xx
		self.diff_red0_xy = diff_red_xy
		self.diff_red0_yy = diff_red_yy

		self.diff_green0_xx = diff_green_xx
		self.diff_green0_xy = diff_green_xy
		self.diff_green0_yy = diff_green_yy

		self.diff_blue0_xx = diff_blue_xx
		self.diff_blue0_xy = diff_blue_xy
		self.diff_blue0_yy = diff_blue_yy


	def forward(self, input):	

		output = self.net(input)

		red = torch.narrow(output,1,0,1) #[1x1xHxW] 
		red = torch.squeeze(red) #[HxW]

		green = torch.narrow(output,1,1,1) #[1x1xHxW] 
		green = torch.squeeze(green) #[HxW]

		blue = torch.narrow(output,1,2,1) #[1x1xHxW] 
		blue = torch.squeeze(blue) #[HxW]

		diff_red_x = derivative_central_x_greylevel(red,dtype)     #[HxW]
		diff_green_x = derivative_central_x_greylevel(green,dtype) #[HxW]
		diff_blue_x = derivative_central_x_greylevel(blue,dtype)   #[HxW]

		diff_red_y = derivative_central_y_greylevel(red,dtype)     #[HxW]
		diff_green_y = derivative_central_y_greylevel(green,dtype) #[HxW]
		diff_blue_y = derivative_central_y_greylevel(blue,dtype)   #[HxW]

		diff_red_xx = derivative_xx_greylevel(red,dtype) #[HxW]
		diff_red_xy = derivative_xy_greylevel(red,dtype) #[HxW]
		diff_red_yy = derivative_yy_greylevel(red,dtype) #[HxW]

		diff_green_xx = derivative_xx_greylevel(green,dtype) #[HxW]
		diff_green_xy = derivative_xy_greylevel(green,dtype) #[HxW]
		diff_green_yy = derivative_yy_greylevel(green,dtype) #[HxW]

		diff_blue_xx = derivative_xx_greylevel(blue,dtype) #[HxW]
		diff_blue_xy = derivative_xy_greylevel(blue,dtype) #[HxW]
		diff_blue_yy = derivative_yy_greylevel(blue,dtype) #[HxW]

		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		covariant_derivative_red_x = diff_red_x - torch.mul(torch.div(self.diff_red0_x,epsilon+self.red0),red)
		covariant_derivative_red_y = diff_red_y - torch.mul(torch.div(self.diff_red0_y,epsilon+self.red0),red)

		covariant_derivative_green_x = diff_green_x - torch.mul(torch.div(self.diff_green0_x,epsilon+self.green0),green)
		covariant_derivative_green_y = diff_green_y - torch.mul(torch.div(self.diff_green0_y,epsilon+self.green0),green)

		covariant_derivative_blue_x = diff_blue_x - torch.mul(torch.div(self.diff_blue0_x,epsilon+self.blue0),blue)
		covariant_derivative_blue_y = diff_blue_y - torch.mul(torch.div(self.diff_blue0_y,epsilon+self.blue0),blue)


		covariant_derivative_red_xx = diff_red_xx - torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xx) - torch.mul(torch.mul(torch.div(2.,epsilon+self.red0),self.diff_red0_x),diff_red_x)+torch.mul(torch.div(2.*red,epsilon+torch.square(self.red0)),torch.square(self.diff_red0_x))
		covariant_derivative_red_xy = diff_red_xy - torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xy) - torch.mul(torch.div(1.,epsilon+self.red0), torch.mul(diff_red_x,self.diff_red0_y)+ torch.mul(diff_red_y,self.diff_red0_x)) + torch.mul(torch.div(2.*red,epsilon+torch.square(self.red0)),torch.mul(self.diff_red0_x,self.diff_red0_y))
		covariant_derivative_red_yy = diff_red_yy - torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_yy) - torch.mul(torch.mul(torch.div(2.,epsilon+self.red0),self.diff_red0_y),diff_red_y)+torch.mul(torch.div(2.*red,epsilon+torch.square(self.red0)),torch.square(self.diff_red0_y))
					

		covariant_derivative_green_xx = diff_green_xx - torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xx) - torch.mul(torch.mul(torch.div(2.,epsilon+self.green0),self.diff_green0_x),diff_green_x)+torch.mul(torch.div(2.*green,epsilon+torch.square(self.green0)),torch.square(self.diff_green0_x))
		covariant_derivative_green_xy = diff_green_xy - torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xy) - torch.mul(torch.div(1.,epsilon+self.green0), torch.mul(diff_green_x,self.diff_green0_y)+ torch.mul(diff_green_y,self.diff_green0_x)) + torch.mul(torch.div(2.*green,epsilon+torch.square(self.green0)),torch.mul(self.diff_green0_x,self.diff_green0_y))
		covariant_derivative_green_yy = diff_green_yy - torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_yy) - torch.mul(torch.mul(torch.div(2.,epsilon+self.green0),self.diff_green0_y),diff_green_y)+torch.mul(torch.div(2.*green,epsilon+torch.square(self.green0)),torch.square(self.diff_green0_y))
		

		covariant_derivative_blue_xx = diff_blue_xx - torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xx) - torch.mul(torch.mul(torch.div(2.,epsilon+self.blue0),self.diff_blue0_x),diff_blue_x)+torch.mul(torch.div(2.*blue,epsilon+torch.square(self.blue0)),torch.square(self.diff_blue0_x))
		covariant_derivative_blue_xy = diff_blue_xy - torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xy) - torch.mul(torch.div(1.,epsilon+self.blue0), torch.mul(diff_blue_x,self.diff_blue0_y)+ torch.mul(diff_blue_y,self.diff_blue0_x)) + torch.mul(torch.div(2.*blue,epsilon+torch.square(self.blue0)),torch.mul(self.diff_blue0_x,self.diff_blue0_y))
		covariant_derivative_blue_yy = diff_blue_yy - torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_yy) - torch.mul(torch.mul(torch.div(2.,epsilon+self.blue0),self.diff_blue0_y),diff_blue_y)+torch.mul(torch.div(2.*blue,epsilon+torch.square(self.blue0)),torch.square(self.diff_blue0_y))

		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_red_x)+torch.square(covariant_derivative_green_x)+torch.square(covariant_derivative_blue_x)) #[HxW]
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_red_y)+torch.square(covariant_derivative_green_y)+torch.square(covariant_derivative_blue_y)) #[HxW]
		g12 = self.beta*(torch.mul(covariant_derivative_red_x,covariant_derivative_red_y)+torch.mul(covariant_derivative_green_x,covariant_derivative_green_y)+torch.mul(covariant_derivative_blue_x,covariant_derivative_blue_y)) #[HxW]
		detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[HxW]


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_red_x)+torch.square(covariant_derivative_green_x)+torch.square(covariant_derivative_blue_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_red_x,covariant_derivative_red_y)+torch.mul(covariant_derivative_green_x,covariant_derivative_green_y)+torch.mul(covariant_derivative_blue_x,covariant_derivative_blue_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_red_y)+torch.square(covariant_derivative_green_y)+torch.square(covariant_derivative_blue_y))

		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_red_xx)+torch.square(covariant_derivative_green_xx)+torch.square(covariant_derivative_blue_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_red_xy)+torch.square(covariant_derivative_green_xy)+torch.square(covariant_derivative_blue_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_red_yy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_yy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_yy,covariant_derivative_blue_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_red_yy)+torch.square(covariant_derivative_green_yy)+torch.square(covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_yy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_yy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xy,covariant_derivative_blue_xy))
		

		norm_regularizer1 = torch.sqrt(epsilon + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]
	
		norm_regularizer2 = torch.sqrt(epsilon + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]


		return output, norm_regularizer1, norm_regularizer2	





class VectorialTotalVariationRplusRplusRplus3rdorderDeblurring(nn.Module):
	def __init__(self,input_depth, pad, beta, epsilon, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorialTotalVariationRplusRplusRplus3rdorderDeblurring,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon

	def forward(self, input):	

		output = self.net(input)

		red = torch.narrow(output,1,0,1) #[1x1xHxW] 
		red = torch.squeeze(red) #[HxW]

		green = torch.narrow(output,1,1,1) #[1x1xHxW] 
		green = torch.squeeze(green) #[HxW]

		blue = torch.narrow(output,1,2,1) #[1x1xHxW] 
		blue = torch.squeeze(blue) #[HxW]

		diff_red_x = derivative_central_x_greylevel(red,dtype)     #[HxW]
		diff_green_x = derivative_central_x_greylevel(green,dtype) #[HxW]
		diff_blue_x = derivative_central_x_greylevel(blue,dtype)   #[HxW]

		diff_red_y = derivative_central_y_greylevel(red,dtype)     #[HxW]
		diff_green_y = derivative_central_y_greylevel(green,dtype) #[HxW]
		diff_blue_y = derivative_central_y_greylevel(blue,dtype)   #[HxW]

		diff_red_xx = derivative_xx_greylevel(red,dtype) #[HxW]
		diff_red_xy = derivative_xy_greylevel(red,dtype) #[HxW]
		diff_red_yy = derivative_yy_greylevel(red,dtype) #[HxW]

		diff_green_xx = derivative_xx_greylevel(green,dtype) #[HxW]
		diff_green_xy = derivative_xy_greylevel(green,dtype) #[HxW]
		diff_green_yy = derivative_yy_greylevel(green,dtype) #[HxW]

		diff_blue_xx = derivative_xx_greylevel(blue,dtype) #[HxW]
		diff_blue_xy = derivative_xy_greylevel(blue,dtype) #[HxW]
		diff_blue_yy = derivative_yy_greylevel(blue,dtype) #[HxW]

		diff_red_xxx = derivative_xxx_greylevel(red,dtype) #[HxW]
		diff_red_xxy = derivative_xxy_greylevel(red,dtype) #[HxW]
		diff_red_xyy = derivative_xyy_greylevel(red,dtype) #[HxW]
		diff_red_yyy = derivative_yyy_greylevel(red,dtype) #[HxW]

		diff_green_xxx = derivative_xxx_greylevel(green,dtype) #[HxW]
		diff_green_xxy = derivative_xxy_greylevel(green,dtype) #[HxW]
		diff_green_xyy = derivative_xyy_greylevel(green,dtype) #[HxW]
		diff_green_yyy = derivative_yyy_greylevel(green,dtype) #[HxW]

		diff_blue_xxx = derivative_xxx_greylevel(blue,dtype) #[HxW]
		diff_blue_xxy = derivative_xxy_greylevel(blue,dtype) #[HxW]
		diff_blue_xyy = derivative_xyy_greylevel(blue,dtype) #[HxW]
		diff_blue_yyy = derivative_yyy_greylevel(blue,dtype) #[HxW]


		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		covariant_derivative_red_x = diff_red_x
		covariant_derivative_red_y = diff_red_y

		covariant_derivative_green_x = diff_green_x
		covariant_derivative_green_y = diff_green_y

		covariant_derivative_blue_x = diff_blue_x
		covariant_derivative_blue_y = diff_blue_y


		covariant_derivative_red_xx = diff_red_xx 
		covariant_derivative_red_xy = diff_red_xy 
		covariant_derivative_red_yy = diff_red_yy
					

		covariant_derivative_green_xx = diff_green_xx 
		covariant_derivative_green_xy = diff_green_xy 
		covariant_derivative_green_yy = diff_green_yy 
		
		covariant_derivative_blue_xx = diff_blue_xx 
		covariant_derivative_blue_xy = diff_blue_xy 
		covariant_derivative_blue_yy = diff_blue_yy 

		covariant_derivative_red_xxx = diff_red_xxx
		covariant_derivative_red_xxy = diff_red_xxy 
		covariant_derivative_red_xyy = diff_red_xyy 
		covariant_derivative_red_yyy = diff_red_yyy 

		covariant_derivative_green_xxx = diff_green_xxx
		covariant_derivative_green_xxy = diff_green_xxy 
		covariant_derivative_green_xyy = diff_green_xyy 
		covariant_derivative_green_yyy = diff_green_yyy 

		covariant_derivative_blue_xxx = diff_blue_xxx 
		covariant_derivative_blue_xxy = diff_blue_xxy 
		covariant_derivative_blue_xyy = diff_blue_xyy 
		covariant_derivative_blue_yyy = diff_blue_yyy 

		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) 
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) 
		g12 = torch.zeros((self.height,self.width)).type(torch.cuda.FloatTensor)
		detg = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) 


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_red_x)+torch.square(covariant_derivative_green_x)+torch.square(covariant_derivative_blue_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_red_x,covariant_derivative_red_y)+torch.mul(covariant_derivative_green_x,covariant_derivative_green_y)+torch.mul(covariant_derivative_blue_x,covariant_derivative_blue_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_red_y)+torch.square(covariant_derivative_green_y)+torch.square(covariant_derivative_blue_y))


		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_red_xx)+torch.square(covariant_derivative_green_xx)+torch.square(covariant_derivative_blue_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_red_xy)+torch.square(covariant_derivative_green_xy)+torch.square(covariant_derivative_blue_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_red_yy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_yy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_yy,covariant_derivative_blue_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_red_yy)+torch.square(covariant_derivative_green_yy)+torch.square(covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_yy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_yy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xy,covariant_derivative_blue_xy))
						

		norm_regularizer3_squared = torch.mul(torch.pow(invg11,3),torch.square(covariant_derivative_red_xxx)+torch.square(covariant_derivative_green_xxx)+torch.square(covariant_derivative_blue_xxx)) \
									+ torch.mul(torch.pow(invg22,3),torch.square(covariant_derivative_red_yyy)+torch.square(covariant_derivative_green_yyy)+torch.square(covariant_derivative_blue_yyy)) \
									+2.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_red_xxx,covariant_derivative_red_yyy)+torch.mul(covariant_derivative_green_xxx,covariant_derivative_green_yyy)+torch.mul(covariant_derivative_blue_xxx,covariant_derivative_blue_yyy)) \
									+6.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_red_xxy,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_xxy,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_xxy,covariant_derivative_blue_xyy)) \
									+12.*torch.mul(torch.mul(torch.mul(invg11,invg22),invg12),torch.mul(covariant_derivative_red_xxy,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_xxy,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_xxy,covariant_derivative_blue_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg11),invg12),torch.mul(covariant_derivative_red_xxx,covariant_derivative_red_xxy)+torch.mul(covariant_derivative_green_xxx,covariant_derivative_green_xxy)+torch.mul(covariant_derivative_blue_xxx,covariant_derivative_blue_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg22),invg12),torch.mul(covariant_derivative_red_yyy,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_yyy,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_yyy,covariant_derivative_blue_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.mul(covariant_derivative_red_xxx,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_xxx,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_xxx,covariant_derivative_blue_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.square(covariant_derivative_red_xxy)+torch.square(covariant_derivative_green_xxy)+torch.square(covariant_derivative_blue_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.mul(covariant_derivative_red_yyy,covariant_derivative_red_xxy)+torch.mul(covariant_derivative_green_yyy,covariant_derivative_green_xxy)+torch.mul(covariant_derivative_blue_yyy,covariant_derivative_blue_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.square(covariant_derivative_red_xyy)+torch.square(covariant_derivative_green_xyy)+torch.square(covariant_derivative_blue_xyy)) \
									+3.*torch.mul(torch.mul(torch.square(invg11),invg22),torch.square(covariant_derivative_red_xxy)+torch.square(covariant_derivative_green_xxy)+torch.square(covariant_derivative_blue_xxy)) \
									+3.*torch.mul(torch.mul(torch.square(invg22),invg11),torch.square(covariant_derivative_red_xyy)+torch.square(covariant_derivative_green_xyy)+torch.square(covariant_derivative_blue_xyy))

		norm_regularizer1 = torch.sqrt(epsilon + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]

		norm_regularizer2 = torch.sqrt(epsilon + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]

		norm_regularizer3 = torch.sqrt(epsilon + norm_regularizer3_squared) #[HxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		

		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 


class VectorBundleTotalVariationRplus3rdorderDeblurring(nn.Module):
	def __init__(self,input_depth, pad, img, diff_img_x, diff_img_y,diff_img_xx,diff_img_xy,diff_img_yy, diff_img_xxx, diff_img_xxy, diff_img_xyy, diff_img_yyy, beta, epsilon, epsilon2, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorBundleTotalVariationRplus3rdorderDeblurring,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon
		self.epsilon2 = epsilon2

		self.img0 = img

		self.diff_img0_x = diff_img_x
		self.diff_img0_y = diff_img_y

		self.diff_img0_xx = diff_img_xx
		self.diff_img0_xy = diff_img_xy
		self.diff_img0_yy = diff_img_yy

		self.diff_img0_xxx = diff_img_xxx
		self.diff_img0_xxy = diff_img_xxy
		self.diff_img0_xyy = diff_img_xyy
		self.diff_img0_yyy = diff_img_yyy

	def forward(self, input):	

		output = self.net(input)

		img = torch.squeeze(output) #[HxW]

		diff_img_x = derivative_central_x_greylevel(img,dtype)     #[HxW]
		diff_img_y = derivative_central_y_greylevel(img,dtype)     #[HxW]


		diff_img_xx = derivative_xx_greylevel(img,dtype) #[HxW]
		diff_img_xy = derivative_xy_greylevel(img,dtype) #[HxW]
		diff_img_yy = derivative_yy_greylevel(img,dtype) #[HxW]

		diff_img_xxx = derivative_xxx_greylevel(img,dtype) #[HxW]
		diff_img_xxy = derivative_xxy_greylevel(img,dtype) #[HxW]
		diff_img_xyy = derivative_xyy_greylevel(img,dtype) #[HxW]
		diff_img_yyy = derivative_yyy_greylevel(img,dtype) #[HxW]

		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)
		epsilon2 = self.epsilon2*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		covariant_derivative_img_x = diff_img_x - torch.mul(torch.div(self.diff_img0_x,epsilon+self.img0),img)
		covariant_derivative_img_y = diff_img_y - torch.mul(torch.div(self.diff_img0_y,epsilon+self.img0),img)

		covariant_derivative_img_xx = diff_img_xx - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xx) - torch.mul(torch.mul(torch.div(2.,epsilon+self.img0),self.diff_img0_x),diff_img_x)+torch.mul(torch.div(2.*img,epsilon+torch.square(self.img0)),torch.square(self.diff_img0_x))
		covariant_derivative_img_xy = diff_img_xy - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xy) - torch.mul(torch.div(1.,epsilon+self.img0), torch.mul(diff_img_x,self.diff_img0_y)+ torch.mul(diff_img_y,self.diff_img0_x)) + torch.mul(torch.div(2.*img,epsilon+torch.square(self.img0)),torch.mul(self.diff_img0_x,self.diff_img0_y))
		covariant_derivative_img_yy = diff_img_yy - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_yy) - torch.mul(torch.mul(torch.div(2.,epsilon+self.img0),self.diff_img0_y),diff_img_y)+torch.mul(torch.div(2.*img,epsilon+torch.square(self.img0)),torch.square(self.diff_img0_y))
	
		covariant_derivative_img_xxx = diff_img_xxx \
									   - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xxx) \
									   - 3.*torch.mul(torch.div(1.,epsilon+self.img0), torch.mul(self.diff_img0_x,diff_img_xx)) \
									   + 6.*torch.mul(torch.div(img,epsilon+torch.square(self.img0)),torch.mul(self.diff_img0_xx,self.diff_img0_x)) \
									   - 3.*torch.mul(torch.div(1.,epsilon+self.img0),torch.mul(self.diff_img0_xx,diff_img_x)) \
									   - 6.*(torch.mul(torch.div(img,epsilon+torch.pow(self.img0,3)),torch.pow(self.diff_img0_x,3))) \
									   + 6.*torch.mul(torch.div(1.,epsilon+torch.square(self.img0)),torch.mul(torch.square(self.diff_img0_x),diff_img_x))
		

		covariant_derivative_img_xxy = diff_img_xxy \
		                               - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xxy) \
		                               - torch.mul(torch.div(1.,epsilon+self.img0),2.*torch.mul(self.diff_img0_x,diff_img_xy)+torch.mul(self.diff_img0_y,diff_img_xx)) \
									   + 2.*torch.mul(torch.div(img,epsilon+torch.square(self.img0)),2.*torch.mul(self.diff_img0_xy,self.diff_img0_x)+torch.mul(self.diff_img0_xx,self.diff_img0_y)) \
									   - torch.mul(torch.div(1.,epsilon+self.img0),2.*torch.mul(self.diff_img0_xy,diff_img_x)+torch.mul(self.diff_img0_xx,diff_img_y)) \
									   - 6.*torch.mul(torch.div(img,epsilon+torch.pow(self.img0,3)),torch.mul(self.diff_img0_y,torch.square(self.diff_img0_x))) \
									   + 2.*torch.mul(torch.div(1.,epsilon+torch.square(self.img0)),2.*torch.mul(self.diff_img0_y,torch.mul(self.diff_img0_x,diff_img_x))+torch.mul(torch.square(self.diff_img0_x),diff_img_y))


		covariant_derivative_img_xyy = diff_img_xyy \
									   - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xyy) \
									   - torch.mul(torch.div(1.,epsilon+self.img0),2.*torch.mul(self.diff_img0_y,diff_img_xy)+torch.mul(self.diff_img0_x,diff_img_yy)) \
									   + 2.*torch.mul(torch.div(img,epsilon+torch.square(self.img0)),2.*torch.mul(self.diff_img0_xy,self.diff_img0_y)+torch.mul(self.diff_img0_yy,self.diff_img0_x)) \
									   - torch.mul(torch.div(1.,epsilon+self.img0),2.*torch.mul(self.diff_img0_xy,diff_img_y)+torch.mul(self.diff_img0_yy,diff_img_x)) \
									   - 6.*torch.mul(torch.div(img,epsilon+torch.pow(self.img0,3)),torch.mul(self.diff_img0_x,torch.square(self.diff_img0_y))) \
									   + 2.*torch.mul(torch.div(1.,epsilon+torch.square(self.img0)),2.*torch.mul(self.diff_img0_x,torch.mul(self.diff_img0_y,diff_img_y))+torch.mul(torch.square(self.diff_img0_y),diff_img_x))

		covariant_derivative_img_yyy = diff_img_yyy \
									   - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_yyy) \
									   - 3.*torch.mul(torch.div(1.,epsilon+self.img0), torch.mul(self.diff_img0_y,diff_img_yy)) \
									   + 6.*torch.mul(torch.div(img,epsilon+torch.square(self.img0)),torch.mul(self.diff_img0_yy,self.diff_img0_y)) \
									   - 3.*torch.mul(torch.div(1.,epsilon+self.img0),torch.mul(self.diff_img0_yy,diff_img_y)) \
									   - 6.*(torch.mul(torch.div(img,epsilon+torch.pow(self.img0,3)),torch.pow(self.diff_img0_y,3))) \
									   + 6.*torch.mul(torch.div(1.,epsilon+torch.square(self.img0)),torch.mul(torch.square(self.diff_img0_y),diff_img_y))

		
		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_img_x)) #[HxW]
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_img_y)) #[HxW]
		g12 = self.beta*(torch.mul(covariant_derivative_img_x,covariant_derivative_img_y)) #[HxW]
		detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[HxW]


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_img_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_img_x,covariant_derivative_img_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_img_y))


		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_img_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_img_xx,covariant_derivative_img_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_img_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_img_yy,covariant_derivative_img_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_img_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_img_xx,covariant_derivative_img_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_img_xy,covariant_derivative_img_xy))
						

		norm_regularizer3_squared = torch.mul(torch.pow(invg11,3),torch.square(covariant_derivative_img_xxx)) \
									+ torch.mul(torch.pow(invg22,3),torch.square(covariant_derivative_img_yyy)) \
									+2.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_yyy))\
									+6.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_img_xxy,covariant_derivative_img_xyy)) \
									+12.*torch.mul(torch.mul(torch.mul(invg11,invg22),invg12),torch.mul(covariant_derivative_img_xxy,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg11),invg12),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg22),invg12),torch.mul(covariant_derivative_img_yyy,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.square(covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.mul(covariant_derivative_img_yyy,covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.square(covariant_derivative_img_xyy)) \
									+3.*torch.mul(torch.mul(torch.square(invg11),invg22),torch.square(covariant_derivative_img_xxy)) \
									+3.*torch.mul(torch.mul(torch.square(invg22),invg11),torch.square(covariant_derivative_img_xyy))

		norm_regularizer1 = torch.sqrt(epsilon2 + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]

		norm_regularizer2 = torch.sqrt(epsilon2 + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]

		norm_regularizer3 = torch.sqrt(epsilon2 + norm_regularizer3_squared) #[HxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		

		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 


class VectorBundleTotalVariationRplus3rdorderDeblurring2(nn.Module):
	def __init__(self,input_depth, pad, img, diff_img_x, diff_img_y,diff_img_xx,diff_img_xy,diff_img_yy, diff_img_xxx, diff_img_xxy, diff_img_xyy, diff_img_yyy, beta, epsilon, epsilon2, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorBundleTotalVariationRplus3rdorderDeblurring2,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon
		self.epsilon2 = epsilon2

		self.img0 = img

		self.diff_img0_x = diff_img_x
		self.diff_img0_y = diff_img_y

		self.diff_img0_xx = diff_img_xx
		self.diff_img0_xy = diff_img_xy
		self.diff_img0_yy = diff_img_yy

		self.diff_img0_xxx = diff_img_xxx
		self.diff_img0_xxy = diff_img_xxy
		self.diff_img0_xyy = diff_img_xyy
		self.diff_img0_yyy = diff_img_yyy

	def forward(self, input):	

		output = self.net(input)

		img = torch.squeeze(output) #[HxW]

		diff_img_x = derivative_central_x_greylevel(img,dtype)     #[HxW]
		diff_img_y = derivative_central_y_greylevel(img,dtype)     #[HxW]


		diff_img_xx = derivative_xx_greylevel(img,dtype) #[HxW]
		diff_img_xy = derivative_xy_greylevel(img,dtype) #[HxW]
		diff_img_yy = derivative_yy_greylevel(img,dtype) #[HxW]

		diff_img_xxx = derivative_xxx_greylevel(img,dtype) #[HxW]
		diff_img_xxy = derivative_xxy_greylevel(img,dtype) #[HxW]
		diff_img_xyy = derivative_xyy_greylevel(img,dtype) #[HxW]
		diff_img_yyy = derivative_yyy_greylevel(img,dtype) #[HxW]

		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)
		epsilon2 = self.epsilon2*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		covariant_derivative_img_x = diff_img_x + torch.mul(torch.div(self.diff_img0_x,epsilon+self.img0),img)
		covariant_derivative_img_y = diff_img_y + torch.mul(torch.div(self.diff_img0_y,epsilon+self.img0),img)

		covariant_derivative_img_xx = diff_img_xx + torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xx) + torch.mul(torch.mul(torch.div(2.,epsilon+self.img0),self.diff_img0_x),diff_img_x)
		covariant_derivative_img_xy = diff_img_xy + torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xy) + torch.mul(torch.div(1.,epsilon+self.img0), torch.mul(diff_img_x,self.diff_img0_y)+ torch.mul(diff_img_y,self.diff_img0_x))
		covariant_derivative_img_yy = diff_img_yy + torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_yy) + torch.mul(torch.mul(torch.div(2.,epsilon+self.img0),self.diff_img0_y),diff_img_y)
	
		covariant_derivative_img_xxx = diff_img_xxx \
									   + torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xxx) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.img0), torch.mul(self.diff_img0_x,diff_img_xx)) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.img0),torch.mul(self.diff_img0_xx,diff_img_x))
									   
				
		covariant_derivative_img_xxy = diff_img_xxy \
		                               + torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xxy) \
		                               + torch.mul(torch.div(1.,epsilon+self.img0),2.*torch.mul(self.diff_img0_x,diff_img_xy)+torch.mul(self.diff_img0_y,diff_img_xx)) \
									   + torch.mul(torch.div(1.,epsilon+self.img0),2.*torch.mul(self.diff_img0_xy,diff_img_x)+torch.mul(self.diff_img0_xx,diff_img_y))
									   

		covariant_derivative_img_xyy = diff_img_xyy \
									   + torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xyy) \
									   + torch.mul(torch.div(1.,epsilon+self.img0),2.*torch.mul(self.diff_img0_y,diff_img_xy)+torch.mul(self.diff_img0_x,diff_img_yy)) \
									   + torch.mul(torch.div(1.,epsilon+self.img0),2.*torch.mul(self.diff_img0_xy,diff_img_y)+torch.mul(self.diff_img0_yy,diff_img_x))
									   
		

		covariant_derivative_img_yyy = diff_img_yyy \
									   + torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_yyy) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.img0), torch.mul(self.diff_img0_y,diff_img_yy)) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.img0),torch.mul(self.diff_img0_yy,diff_img_y)) \
									 
		
		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_img_x)) #[HxW]
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_img_y)) #[HxW]
		g12 = self.beta*(torch.mul(covariant_derivative_img_x,covariant_derivative_img_y)) #[HxW]
		detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[HxW]


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_img_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_img_x,covariant_derivative_img_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_img_y))


		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_img_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_img_xx,covariant_derivative_img_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_img_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_img_yy,covariant_derivative_img_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_img_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_img_xx,covariant_derivative_img_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_img_xy,covariant_derivative_img_xy))
						

		norm_regularizer3_squared = torch.mul(torch.pow(invg11,3),torch.square(covariant_derivative_img_xxx)) \
									+ torch.mul(torch.pow(invg22,3),torch.square(covariant_derivative_img_yyy)) \
									+2.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_yyy))\
									+6.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_img_xxy,covariant_derivative_img_xyy)) \
									+12.*torch.mul(torch.mul(torch.mul(invg11,invg22),invg12),torch.mul(covariant_derivative_img_xxy,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg11),invg12),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg22),invg12),torch.mul(covariant_derivative_img_yyy,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.square(covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.mul(covariant_derivative_img_yyy,covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.square(covariant_derivative_img_xyy)) \
									+3.*torch.mul(torch.mul(torch.square(invg11),invg22),torch.square(covariant_derivative_img_xxy)) \
									+3.*torch.mul(torch.mul(torch.square(invg22),invg11),torch.square(covariant_derivative_img_xyy))

		norm_regularizer1 = torch.sqrt(epsilon2 + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]

		norm_regularizer2 = torch.sqrt(epsilon2 + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]

		norm_regularizer3 = torch.sqrt(epsilon2 + norm_regularizer3_squared) #[HxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		

		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 



class VectorBundleTotalVariationRplus3rdorderDeblurringTest(nn.Module):
	def __init__(self,input_depth, pad, img, diff_img_x, diff_img_y,diff_img_xx,diff_img_xy,diff_img_yy, diff_img_xxx, diff_img_xxy, diff_img_xyy, diff_img_yyy, beta, epsilon, epsilon2, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorBundleTotalVariationRplus3rdorderDeblurringTest,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon
		self.epsilon2 = epsilon2

		self.img0 = img

		self.diff_img0_x = diff_img_x
		self.diff_img0_y = diff_img_y

		self.diff_img0_xx = diff_img_xx
		self.diff_img0_xy = diff_img_xy
		self.diff_img0_yy = diff_img_yy

		self.diff_img0_xxx = diff_img_xxx
		self.diff_img0_xxy = diff_img_xxy
		self.diff_img0_xyy = diff_img_xyy
		self.diff_img0_yyy = diff_img_yyy

	def forward(self, input):	

		NOISE_SIGMA = 2**.5  # sqrt(2), I haven't tests other options
		BLUR_TYPE = 'uniform_blur'  # 'gauss_blur' or 'uniform_blur' that the two only options
		GRAY_SCALE = False  # if gray scale is False means we have rgb image, the psnr will be compared on Y. ch.
	                    # if gray scale is True it will turn rgb to gray scale
		USE_FOURIER = False

		H = get_h(1, BLUR_TYPE, USE_FOURIER, torch.cuda.FloatTensor)

		output = self.net(input)
		output = H(output)

		img = torch.squeeze(output) #[HxW]

		diff_img_x = derivative_central_x_greylevel(img,dtype)     #[HxW]
		diff_img_y = derivative_central_y_greylevel(img,dtype)     #[HxW]


		diff_img_xx = derivative_xx_greylevel(img,dtype) #[HxW]
		diff_img_xy = derivative_xy_greylevel(img,dtype) #[HxW]
		diff_img_yy = derivative_yy_greylevel(img,dtype) #[HxW]

		diff_img_xxx = derivative_xxx_greylevel(img,dtype) #[HxW]
		diff_img_xxy = derivative_xxy_greylevel(img,dtype) #[HxW]
		diff_img_xyy = derivative_xyy_greylevel(img,dtype) #[HxW]
		diff_img_yyy = derivative_yyy_greylevel(img,dtype) #[HxW]

		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)
		epsilon2 = self.epsilon2*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		covariant_derivative_img_x = diff_img_x - torch.mul(torch.div(self.diff_img0_x,epsilon+self.img0),img)
		covariant_derivative_img_y = diff_img_y - torch.mul(torch.div(self.diff_img0_y,epsilon+self.img0),img)

		covariant_derivative_img_xx = diff_img_xx - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xx) - torch.mul(torch.mul(torch.div(2.,epsilon+self.img0),self.diff_img0_x),diff_img_x)+torch.mul(torch.div(2.*img,epsilon+torch.square(self.img0)),torch.square(self.diff_img0_x))
		covariant_derivative_img_xy = diff_img_xy - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xy) - torch.mul(torch.div(1.,epsilon+self.img0), torch.mul(diff_img_x,self.diff_img0_y)+ torch.mul(diff_img_y,self.diff_img0_x)) + torch.mul(torch.div(2.*img,epsilon+torch.square(self.img0)),torch.mul(self.diff_img0_x,self.diff_img0_y))
		covariant_derivative_img_yy = diff_img_yy - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_yy) - torch.mul(torch.mul(torch.div(2.,epsilon+self.img0),self.diff_img0_y),diff_img_y)+torch.mul(torch.div(2.*img,epsilon+torch.square(self.img0)),torch.square(self.diff_img0_y))
	
		covariant_derivative_img_xxx = diff_img_xxx \
									   - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xxx) \
									   - 3.*torch.mul(torch.div(1.,epsilon+self.img0), torch.mul(self.diff_img0_x,diff_img_xx)) \
									   + 6.*torch.mul(torch.div(img,epsilon+torch.square(self.img0)),torch.mul(self.diff_img0_xx,self.diff_img0_x)) \
									   - 3.*torch.mul(torch.div(1.,epsilon+self.img0),torch.mul(self.diff_img0_xx,diff_img_x)) \
									   - 6.*(torch.mul(torch.div(img,epsilon+torch.pow(self.img0,3)),torch.pow(self.diff_img0_x,3))) \
									   + 6.*torch.mul(torch.div(1.,epsilon+torch.square(self.img0)),torch.mul(torch.square(self.diff_img0_x),diff_img_x))
		

		covariant_derivative_img_xxy = diff_img_xxy \
		                               - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xxy) \
		                               - torch.mul(torch.div(1.,epsilon+self.img0),2.*torch.mul(self.diff_img0_x,diff_img_xy)+torch.mul(self.diff_img0_y,diff_img_xx)) \
									   + 2.*torch.mul(torch.div(img,epsilon+torch.square(self.img0)),2.*torch.mul(self.diff_img0_xy,self.diff_img0_x)+torch.mul(self.diff_img0_xx,self.diff_img0_y)) \
									   - torch.mul(torch.div(1.,epsilon+self.img0),2.*torch.mul(self.diff_img0_xy,diff_img_x)+torch.mul(self.diff_img0_xx,diff_img_y)) \
									   - 6.*torch.mul(torch.div(img,epsilon+torch.pow(self.img0,3)),torch.mul(self.diff_img0_y,torch.square(self.diff_img0_x))) \
									   + 2.*torch.mul(torch.div(1.,epsilon+torch.square(self.img0)),2.*torch.mul(self.diff_img0_y,torch.mul(self.diff_img0_x,diff_img_x))+torch.mul(torch.square(self.diff_img0_x),diff_img_y))


		covariant_derivative_img_xyy = diff_img_xyy \
									   - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_xyy) \
									   - torch.mul(torch.div(1.,epsilon+self.img0),2.*torch.mul(self.diff_img0_y,diff_img_xy)+torch.mul(self.diff_img0_x,diff_img_yy)) \
									   + 2.*torch.mul(torch.div(img,epsilon+torch.square(self.img0)),2.*torch.mul(self.diff_img0_xy,self.diff_img0_y)+torch.mul(self.diff_img0_yy,self.diff_img0_x)) \
									   - torch.mul(torch.div(1.,epsilon+self.img0),2.*torch.mul(self.diff_img0_xy,diff_img_y)+torch.mul(self.diff_img0_yy,diff_img_x)) \
									   - 6.*torch.mul(torch.div(img,epsilon+torch.pow(self.img0,3)),torch.mul(self.diff_img0_x,torch.square(self.diff_img0_y))) \
									   + 2.*torch.mul(torch.div(1.,epsilon+torch.square(self.img0)),2.*torch.mul(self.diff_img0_x,torch.mul(self.diff_img0_y,diff_img_y))+torch.mul(torch.square(self.diff_img0_y),diff_img_x))

		covariant_derivative_img_yyy = diff_img_yyy \
									   - torch.mul(torch.div(img,epsilon+self.img0),self.diff_img0_yyy) \
									   - 3.*torch.mul(torch.div(1.,epsilon+self.img0), torch.mul(self.diff_img0_y,diff_img_yy)) \
									   + 6.*torch.mul(torch.div(img,epsilon+torch.square(self.img0)),torch.mul(self.diff_img0_yy,self.diff_img0_y)) \
									   - 3.*torch.mul(torch.div(1.,epsilon+self.img0),torch.mul(self.diff_img0_yy,diff_img_y)) \
									   - 6.*(torch.mul(torch.div(img,epsilon+torch.pow(self.img0,3)),torch.pow(self.diff_img0_y,3))) \
									   + 6.*torch.mul(torch.div(1.,epsilon+torch.square(self.img0)),torch.mul(torch.square(self.diff_img0_y),diff_img_y))

		
		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_img_x)) #[HxW]
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_img_y)) #[HxW]
		g12 = self.beta*(torch.mul(covariant_derivative_img_x,covariant_derivative_img_y)) #[HxW]
		detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[HxW]


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_img_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_img_x,covariant_derivative_img_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_img_y))


		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_img_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_img_xx,covariant_derivative_img_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_img_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_img_yy,covariant_derivative_img_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_img_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_img_xx,covariant_derivative_img_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_img_xy,covariant_derivative_img_xy))
						

		norm_regularizer3_squared = torch.mul(torch.pow(invg11,3),torch.square(covariant_derivative_img_xxx)) \
									+ torch.mul(torch.pow(invg22,3),torch.square(covariant_derivative_img_yyy)) \
									+2.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_yyy))\
									+6.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_img_xxy,covariant_derivative_img_xyy)) \
									+12.*torch.mul(torch.mul(torch.mul(invg11,invg22),invg12),torch.mul(covariant_derivative_img_xxy,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg11),invg12),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg22),invg12),torch.mul(covariant_derivative_img_yyy,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.mul(covariant_derivative_img_xxx,covariant_derivative_img_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.square(covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.mul(covariant_derivative_img_yyy,covariant_derivative_img_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.square(covariant_derivative_img_xyy)) \
									+3.*torch.mul(torch.mul(torch.square(invg11),invg22),torch.square(covariant_derivative_img_xxy)) \
									+3.*torch.mul(torch.mul(torch.square(invg22),invg11),torch.square(covariant_derivative_img_xyy))

		norm_regularizer1 = torch.sqrt(epsilon2 + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]

		norm_regularizer2 = torch.sqrt(epsilon2 + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]

		norm_regularizer3 = torch.sqrt(epsilon2 + norm_regularizer3_squared) #[HxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		

		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 		


class VectorBundleTotalVariationRplusRplusRplus3rdorderDeblurring(nn.Module):
	def __init__(self,input_depth, pad, red, green, blue, diff_red_x,diff_green_x,diff_blue_x, diff_red_y,diff_green_y,diff_blue_y,diff_red_xx,diff_red_xy,diff_red_yy,diff_green_xx,diff_green_xy,diff_green_yy,diff_blue_xx,diff_blue_xy,diff_blue_yy, diff_red_xxx, diff_red_xxy, diff_red_xyy, diff_red_yyy, diff_green_xxx, diff_green_xxy, diff_green_xyy, diff_green_yyy, diff_blue_xxx, diff_blue_xxy, diff_blue_xyy, diff_blue_yyy, gamma, beta, epsilon, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorBundleTotalVariationRplusRplusRplus3rdorderDeblurring,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.gamma = gamma
		self.epsilon = epsilon

		self.red0 = red
		self.green0 = green
		self.blue0 = blue

		self.diff_red0_x = diff_red_x
		self.diff_green0_x = diff_green_x
		self.diff_blue0_x = diff_blue_x
		self.diff_red0_y = diff_red_y
		self.diff_green0_y = diff_green_y
		self.diff_blue0_y = diff_blue_y

		self.diff_red0_xx = diff_red_xx
		self.diff_red0_xy = diff_red_xy
		self.diff_red0_yy = diff_red_yy

		self.diff_green0_xx = diff_green_xx
		self.diff_green0_xy = diff_green_xy
		self.diff_green0_yy = diff_green_yy

		self.diff_blue0_xx = diff_blue_xx
		self.diff_blue0_xy = diff_blue_xy
		self.diff_blue0_yy = diff_blue_yy

		self.diff_red0_xxx = diff_red_xxx
		self.diff_red0_xxy = diff_red_xxy
		self.diff_red0_xyy = diff_red_xyy
		self.diff_red0_yyy = diff_red_yyy

		self.diff_green0_xxx = diff_green_xxx
		self.diff_green0_xxy = diff_green_xxy
		self.diff_green0_xyy = diff_green_xyy
		self.diff_green0_yyy = diff_green_yyy

		self.diff_blue0_xxx = diff_blue_xxx
		self.diff_blue0_xxy = diff_blue_xxy
		self.diff_blue0_xyy = diff_blue_xyy
		self.diff_blue0_yyy = diff_blue_yyy


	def forward(self, input):	

		output = self.net(input)

		red = torch.narrow(output,1,0,1) #[1x1xHxW] 
		red = torch.squeeze(red) #[HxW]

		green = torch.narrow(output,1,1,1) #[1x1xHxW] 
		green = torch.squeeze(green) #[HxW]

		blue = torch.narrow(output,1,2,1) #[1x1xHxW] 
		blue = torch.squeeze(blue) #[HxW]

		diff_red_x = derivative_central_x_greylevel(red,dtype)     #[HxW]
		diff_green_x = derivative_central_x_greylevel(green,dtype) #[HxW]
		diff_blue_x = derivative_central_x_greylevel(blue,dtype)   #[HxW]

		diff_red_y = derivative_central_y_greylevel(red,dtype)     #[HxW]
		diff_green_y = derivative_central_y_greylevel(green,dtype) #[HxW]
		diff_blue_y = derivative_central_y_greylevel(blue,dtype)   #[HxW]

		diff_red_xx = derivative_xx_greylevel(red,dtype) #[HxW]
		diff_red_xy = derivative_xy_greylevel(red,dtype) #[HxW]
		diff_red_yy = derivative_yy_greylevel(red,dtype) #[HxW]

		diff_green_xx = derivative_xx_greylevel(green,dtype) #[HxW]
		diff_green_xy = derivative_xy_greylevel(green,dtype) #[HxW]
		diff_green_yy = derivative_yy_greylevel(green,dtype) #[HxW]

		diff_blue_xx = derivative_xx_greylevel(blue,dtype) #[HxW]
		diff_blue_xy = derivative_xy_greylevel(blue,dtype) #[HxW]
		diff_blue_yy = derivative_yy_greylevel(blue,dtype) #[HxW]

		diff_red_xxx = derivative_xxx_greylevel(red,dtype) #[HxW]
		diff_red_xxy = derivative_xxy_greylevel(red,dtype) #[HxW]
		diff_red_xyy = derivative_xyy_greylevel(red,dtype) #[HxW]
		diff_red_yyy = derivative_yyy_greylevel(red,dtype) #[HxW]

		diff_green_xxx = derivative_xxx_greylevel(green,dtype) #[HxW]
		diff_green_xxy = derivative_xxy_greylevel(green,dtype) #[HxW]
		diff_green_xyy = derivative_xyy_greylevel(green,dtype) #[HxW]
		diff_green_yyy = derivative_yyy_greylevel(green,dtype) #[HxW]

		diff_blue_xxx = derivative_xxx_greylevel(blue,dtype) #[HxW]
		diff_blue_xxy = derivative_xxy_greylevel(blue,dtype) #[HxW]
		diff_blue_xyy = derivative_xyy_greylevel(blue,dtype) #[HxW]
		diff_blue_yyy = derivative_yyy_greylevel(blue,dtype) #[HxW]


		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		covariant_derivative_red_x = diff_red_x - self.gamma*(torch.mul(torch.div(self.diff_red0_x,epsilon+self.red0),red))
		covariant_derivative_red_y = diff_red_y - self.gamma*(torch.mul(torch.div(self.diff_red0_y,epsilon+self.red0),red))

		covariant_derivative_green_x = diff_green_x - self.gamma*(torch.mul(torch.div(self.diff_green0_x,epsilon+self.green0),green))
		covariant_derivative_green_y = diff_green_y - self.gamma*(torch.mul(torch.div(self.diff_green0_y,epsilon+self.green0),green))

		covariant_derivative_blue_x = diff_blue_x - self.gamma*(torch.mul(torch.div(self.diff_blue0_x,epsilon+self.blue0),blue))
		covariant_derivative_blue_y = diff_blue_y - self.gamma*(torch.mul(torch.div(self.diff_blue0_y,epsilon+self.blue0),blue))


		covariant_derivative_red_xx = diff_red_xx - self.gamma*(torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xx)) - torch.mul(torch.mul(torch.div(2.*self.gamma,epsilon+self.red0),self.diff_red0_x),diff_red_x)+torch.mul(torch.div((self.gamma+self.gamma**2)*red,epsilon+torch.square(self.red0)),torch.square(self.diff_red0_x))
		covariant_derivative_red_xy = diff_red_xy - self.gamma*(torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xy)) - torch.mul(torch.div(self.gamma,epsilon+self.red0), torch.mul(diff_red_x,self.diff_red0_y)+ torch.mul(diff_red_y,self.diff_red0_x)) + torch.mul(torch.div((self.gamma+self.gamma**2)*red,epsilon+torch.square(self.red0)),torch.mul(self.diff_red0_x,self.diff_red0_y))
		covariant_derivative_red_yy = diff_red_yy - self.gamma*(torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_yy)) - torch.mul(torch.mul(torch.div(2.*self.gamma,epsilon+self.red0),self.diff_red0_y),diff_red_y)+torch.mul(torch.div((self.gamma+self.gamma**2)*red,epsilon+torch.square(self.red0)),torch.square(self.diff_red0_y))
					

		covariant_derivative_green_xx = diff_green_xx - self.gamma*(torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xx)) - torch.mul(torch.mul(torch.div(2.*self.gamma,epsilon+self.green0),self.diff_green0_x),diff_green_x)+torch.mul(torch.div((self.gamma+self.gamma**2)*green,epsilon+torch.square(self.green0)),torch.square(self.diff_green0_x))
		covariant_derivative_green_xy = diff_green_xy - self.gamma*(torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xy)) - torch.mul(torch.div(self.gamma,epsilon+self.green0), torch.mul(diff_green_x,self.diff_green0_y)+ torch.mul(diff_green_y,self.diff_green0_x)) + torch.mul(torch.div((self.gamma+self.gamma**2)*green,epsilon+torch.square(self.green0)),torch.mul(self.diff_green0_x,self.diff_green0_y))
		covariant_derivative_green_yy = diff_green_yy - self.gamma*(torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_yy)) - torch.mul(torch.mul(torch.div(2.*self.gamma,epsilon+self.green0),self.diff_green0_y),diff_green_y)+torch.mul(torch.div((self.gamma+self.gamma**2)*green,epsilon+torch.square(self.green0)),torch.square(self.diff_green0_y))
		
		covariant_derivative_blue_xx = diff_blue_xx - self.gamma*(torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xx)) - torch.mul(torch.mul(torch.div(2.*self.gamma,epsilon+self.blue0),self.diff_blue0_x),diff_blue_x)+torch.mul(torch.div((self.gamma+self.gamma**2)*blue,epsilon+torch.square(self.blue0)),torch.square(self.diff_blue0_x))
		covariant_derivative_blue_xy = diff_blue_xy - self.gamma*(torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xy)) - torch.mul(torch.div(self.gamma,epsilon+self.blue0), torch.mul(diff_blue_x,self.diff_blue0_y)+ torch.mul(diff_blue_y,self.diff_blue0_x)) + torch.mul(torch.div((self.gamma+self.gamma**2)*blue,epsilon+torch.square(self.blue0)),torch.mul(self.diff_blue0_x,self.diff_blue0_y))
		covariant_derivative_blue_yy = diff_blue_yy - self.gamma*(torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_yy)) - torch.mul(torch.mul(torch.div(2.*self.gamma,epsilon+self.blue0),self.diff_blue0_y),diff_blue_y)+torch.mul(torch.div((self.gamma+self.gamma**2)*blue,epsilon+torch.square(self.blue0)),torch.square(self.diff_blue0_y))

		covariant_derivative_red_xxx = diff_red_xxx \
									   - self.gamma*(torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xxx)) \
									   - 3.*torch.mul(torch.div(self.gamma,epsilon+self.red0), torch.mul(self.diff_red0_x,diff_red_xx)) \
									   + 3.*(self.gamma+self.gamma**2)*torch.mul(torch.div(red,epsilon+torch.square(self.red0)),torch.mul(self.diff_red0_xx,self.diff_red0_x)) \
									   - 3.*torch.mul(torch.div(self.gamma,epsilon+self.red0),torch.mul(self.diff_red0_xx,diff_red_x)) \
									   - (2.+self.gamma)*(self.gamma+self.gamma**2)*(torch.mul(torch.div(red,epsilon+torch.pow(self.red0,3)),torch.pow(self.diff_red0_x,3))) \
									   + 3.*(self.gamma+self.gamma**2)*torch.mul(torch.div(1.,epsilon+torch.square(self.red0)),torch.mul(torch.square(self.diff_red0_x),diff_red_x))
		

		covariant_derivative_red_xxy = diff_red_xxy \
		                               - self.gamma*(torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xxy)) \
		                               - torch.mul(torch.div(self.gamma,epsilon+self.red0),2.*torch.mul(self.diff_red0_x,diff_red_xy)+torch.mul(self.diff_red0_y,diff_red_xx)) \
									   + (self.gamma+self.gamma**2)*torch.mul(torch.div(red,epsilon+torch.square(self.red0)),2.*torch.mul(self.diff_red0_xy,self.diff_red0_x)+torch.mul(self.diff_red0_xx,self.diff_red0_y)) \
									   - torch.mul(torch.div(self.gamma,epsilon+self.red0),2.*torch.mul(self.diff_red0_xy,diff_red_x)+torch.mul(self.diff_red0_xx,diff_red_y)) \
									   - (2.+self.gamma)*(self.gamma+self.gamma**2)*torch.mul(torch.div(red,epsilon+torch.pow(self.red0,3)),torch.mul(self.diff_red0_y,torch.square(self.diff_red0_x))) \
									   + (self.gamma+self.gamma**2)*torch.mul(torch.div(1.,epsilon+torch.square(self.red0)),2.*torch.mul(self.diff_red0_y,torch.mul(self.diff_red0_x,diff_red_x))+torch.mul(torch.square(self.diff_red0_x),diff_red_y))


		covariant_derivative_red_xyy = diff_red_xyy \
									   - self.gamma*(torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xyy)) \
									   - torch.mul(torch.div(self.gamma,epsilon+self.red0),2.*torch.mul(self.diff_red0_y,diff_red_xy)+torch.mul(self.diff_red0_x,diff_red_yy)) \
									   + (self.gamma+self.gamma**2)*torch.mul(torch.div(red,epsilon+torch.square(self.red0)),2.*torch.mul(self.diff_red0_xy,self.diff_red0_y)+torch.mul(self.diff_red0_yy,self.diff_red0_x)) \
									   - torch.mul(torch.div(self.gamma,epsilon+self.red0),2.*torch.mul(self.diff_red0_xy,diff_red_y)+torch.mul(self.diff_red0_yy,diff_red_x)) \
									   - (2.+self.gamma)*(self.gamma+self.gamma**2)*torch.mul(torch.div(red,epsilon+torch.pow(self.red0,3)),torch.mul(self.diff_red0_x,torch.square(self.diff_red0_y))) \
									   + (self.gamma+self.gamma**2)*torch.mul(torch.div(1.,epsilon+torch.square(self.red0)),2.*torch.mul(self.diff_red0_x,torch.mul(self.diff_red0_y,diff_red_y))+torch.mul(torch.square(self.diff_red0_y),diff_red_x))

		covariant_derivative_red_yyy = diff_red_yyy \
									   - self.gamma*(torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_yyy)) \
									   - 3.*torch.mul(torch.div(self.gamma,epsilon+self.red0), torch.mul(self.diff_red0_y,diff_red_yy)) \
									   + 3.*(self.gamma+self.gamma**2)*torch.mul(torch.div(red,epsilon+torch.square(self.red0)),torch.mul(self.diff_red0_yy,self.diff_red0_y)) \
									   - 3.*torch.mul(torch.div(self.gamma,epsilon+self.red0),torch.mul(self.diff_red0_yy,diff_red_y)) \
									   - (2.+self.gamma)*(self.gamma+self.gamma**2)*(torch.mul(torch.div(red,epsilon+torch.pow(self.red0,3)),torch.pow(self.diff_red0_y,3))) \
									   + 3.*(self.gamma+self.gamma**2)*torch.mul(torch.div(1.,epsilon+torch.square(self.red0)),torch.mul(torch.square(self.diff_red0_y),diff_red_y))

	
		covariant_derivative_green_xxx = diff_green_xxx \
									   - self.gamma*(torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xxx)) \
									   - 3.*torch.mul(torch.div(self.gamma,epsilon+self.green0), torch.mul(self.diff_green0_x,diff_green_xx)) \
									   + 3.*(self.gamma+self.gamma**2)*torch.mul(torch.div(green,epsilon+torch.square(self.green0)),torch.mul(self.diff_green0_xx,self.diff_green0_x)) \
									   - 3.*torch.mul(torch.div(self.gamma,epsilon+self.green0),torch.mul(self.diff_green0_xx,diff_green_x)) \
									   - (2.+self.gamma)*(self.gamma+self.gamma**2)*(torch.mul(torch.div(green,epsilon+torch.pow(self.green0,3)),torch.pow(self.diff_green0_x,3))) \
									   + 3.*(self.gamma+self.gamma**2)*torch.mul(torch.div(1.,epsilon+torch.square(self.green0)),torch.mul(torch.square(self.diff_green0_x),diff_green_x))
		

		covariant_derivative_green_xxy = diff_green_xxy \
		                               - self.gamma*(torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xxy)) \
		                               - torch.mul(torch.div(self.gamma,epsilon+self.green0),2.*torch.mul(self.diff_green0_x,diff_green_xy)+torch.mul(self.diff_green0_y,diff_green_xx)) \
									   + (self.gamma+self.gamma**2)*torch.mul(torch.div(green,epsilon+torch.square(self.green0)),2.*torch.mul(self.diff_green0_xy,self.diff_green0_x)+torch.mul(self.diff_green0_xx,self.diff_green0_y)) \
									   - torch.mul(torch.div(self.gamma,epsilon+self.green0),2.*torch.mul(self.diff_green0_xy,diff_green_x)+torch.mul(self.diff_green0_xx,diff_green_y)) \
									   - (2.+self.gamma)*(self.gamma+self.gamma**2)*torch.mul(torch.div(green,epsilon+torch.pow(self.green0,3)),torch.mul(self.diff_green0_y,torch.square(self.diff_green0_x))) \
									   + (self.gamma+self.gamma**2)*torch.mul(torch.div(1.,epsilon+torch.square(self.green0)),2.*torch.mul(self.diff_green0_y,torch.mul(self.diff_green0_x,diff_green_x))+torch.mul(torch.square(self.diff_green0_x),diff_green_y))


		covariant_derivative_green_xyy = diff_green_xyy \
									   - self.gamma*(torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xyy)) \
									   - torch.mul(torch.div(self.gamma,epsilon+self.green0),2.*torch.mul(self.diff_green0_y,diff_green_xy)+torch.mul(self.diff_green0_x,diff_green_yy)) \
									   + (self.gamma+self.gamma**2)*torch.mul(torch.div(green,epsilon+torch.square(self.green0)),2.*torch.mul(self.diff_green0_xy,self.diff_green0_y)+torch.mul(self.diff_green0_yy,self.diff_green0_x)) \
									   - torch.mul(torch.div(self.gamma,epsilon+self.green0),2.*torch.mul(self.diff_green0_xy,diff_green_y)+torch.mul(self.diff_green0_yy,diff_green_x)) \
									   - (2.+self.gamma)*(self.gamma+self.gamma**2)*torch.mul(torch.div(green,epsilon+torch.pow(self.green0,3)),torch.mul(self.diff_green0_x,torch.square(self.diff_green0_y))) \
									   + (self.gamma+self.gamma**2)*torch.mul(torch.div(1.,epsilon+torch.square(self.green0)),2.*torch.mul(self.diff_green0_x,torch.mul(self.diff_green0_y,diff_green_y))+torch.mul(torch.square(self.diff_green0_y),diff_green_x))

		covariant_derivative_green_yyy = diff_green_yyy \
									   - self.gamma*(torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_yyy)) \
									   - 3.*torch.mul(torch.div(self.gamma,epsilon+self.green0), torch.mul(self.diff_green0_y,diff_green_yy)) \
									   + 3.*(self.gamma+self.gamma**2)*torch.mul(torch.div(green,epsilon+torch.square(self.green0)),torch.mul(self.diff_green0_yy,self.diff_green0_y)) \
									   - 3.*torch.mul(torch.div(self.gamma,epsilon+self.green0),torch.mul(self.diff_green0_yy,diff_green_y)) \
									   - (2.+self.gamma)*(self.gamma+self.gamma**2)*(torch.mul(torch.div(green,epsilon+torch.pow(self.green0,3)),torch.pow(self.diff_green0_y,3))) \
									   + 3.*(self.gamma+self.gamma**2)*torch.mul(torch.div(1.,epsilon+torch.square(self.green0)),torch.mul(torch.square(self.diff_green0_y),diff_green_y))
							   


		covariant_derivative_blue_xxx = diff_blue_xxx \
									   - self.gamma*(torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xxx)) \
									   - 3.*torch.mul(torch.div(self.gamma,epsilon+self.blue0), torch.mul(self.diff_blue0_x,diff_blue_xx)) \
									   + 3.*(self.gamma+self.gamma**2)*torch.mul(torch.div(blue,epsilon+torch.square(self.blue0)),torch.mul(self.diff_blue0_xx,self.diff_blue0_x)) \
									   - 3.*torch.mul(torch.div(self.gamma,epsilon+self.blue0),torch.mul(self.diff_blue0_xx,diff_blue_x)) \
									   - (2.+self.gamma)*(self.gamma+self.gamma**2)*(torch.mul(torch.div(blue,epsilon+torch.pow(self.blue0,3)),torch.pow(self.diff_blue0_x,3))) \
									   + 3.*(self.gamma+self.gamma**2)*torch.mul(torch.div(1.,epsilon+torch.square(self.blue0)),torch.mul(torch.square(self.diff_blue0_x),diff_blue_x))
		

		covariant_derivative_blue_xxy = diff_blue_xxy \
		                               - self.gamma*(torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xxy)) \
		                               - torch.mul(torch.div(self.gamma,epsilon+self.blue0),2.*torch.mul(self.diff_blue0_x,diff_blue_xy)+torch.mul(self.diff_blue0_y,diff_blue_xx)) \
									   + (self.gamma+self.gamma**2)*torch.mul(torch.div(blue,epsilon+torch.square(self.blue0)),2.*torch.mul(self.diff_blue0_xy,self.diff_blue0_x)+torch.mul(self.diff_blue0_xx,self.diff_blue0_y)) \
									   - torch.mul(torch.div(self.gamma,epsilon+self.blue0),2.*torch.mul(self.diff_blue0_xy,diff_blue_x)+torch.mul(self.diff_blue0_xx,diff_blue_y)) \
									   - (2.+self.gamma)*(self.gamma+self.gamma**2)*torch.mul(torch.div(blue,epsilon+torch.pow(self.blue0,3)),torch.mul(self.diff_blue0_y,torch.square(self.diff_blue0_x))) \
									   + (self.gamma+self.gamma**2)*torch.mul(torch.div(1.,epsilon+torch.square(self.blue0)),2.*torch.mul(self.diff_blue0_y,torch.mul(self.diff_blue0_x,diff_blue_x))+torch.mul(torch.square(self.diff_blue0_x),diff_blue_y))


		covariant_derivative_blue_xyy = diff_blue_xyy \
									   - self.gamma*(torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xyy)) \
									   - torch.mul(torch.div(self.gamma,epsilon+self.blue0),2.*torch.mul(self.diff_blue0_y,diff_blue_xy)+torch.mul(self.diff_blue0_x,diff_blue_yy)) \
									   + (self.gamma+self.gamma**2)*torch.mul(torch.div(blue,epsilon+torch.square(self.blue0)),2.*torch.mul(self.diff_blue0_xy,self.diff_blue0_y)+torch.mul(self.diff_blue0_yy,self.diff_blue0_x)) \
									   - torch.mul(torch.div(self.gamma,epsilon+self.blue0),2.*torch.mul(self.diff_blue0_xy,diff_blue_y)+torch.mul(self.diff_blue0_yy,diff_blue_x)) \
									   - 6.*torch.mul(torch.div(blue,epsilon+torch.pow(self.blue0,3)),torch.mul(self.diff_blue0_x,torch.square(self.diff_blue0_y))) \
									   + (self.gamma+self.gamma**2)*torch.mul(torch.div(1.,epsilon+torch.square(self.blue0)),2.*torch.mul(self.diff_blue0_x,torch.mul(self.diff_blue0_y,diff_blue_y))+torch.mul(torch.square(self.diff_blue0_y),diff_blue_x))

		covariant_derivative_blue_yyy = diff_blue_yyy \
									   - self.gamma*(torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_yyy)) \
									   - 3.*torch.mul(torch.div(self.gamma,epsilon+self.blue0), torch.mul(self.diff_blue0_y,diff_blue_yy)) \
									   + 3.*(self.gamma+self.gamma**2)*torch.mul(torch.div(blue,epsilon+torch.square(self.blue0)),torch.mul(self.diff_blue0_yy,self.diff_blue0_y)) \
									   - 3.*torch.mul(torch.div(self.gamma,epsilon+self.blue0),torch.mul(self.diff_blue0_yy,diff_blue_y)) \
									   - 6.*(torch.mul(torch.div(blue,epsilon+torch.pow(self.blue0,3)),torch.pow(self.diff_blue0_y,3))) \
									   + 3.*(self.gamma+self.gamma**2)*torch.mul(torch.div(1.,epsilon+torch.square(self.blue0)),torch.mul(torch.square(self.diff_blue0_y),diff_blue_y))
		

		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_red_x)+torch.square(covariant_derivative_green_x)+torch.square(covariant_derivative_blue_x)) #[HxW]
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_red_y)+torch.square(covariant_derivative_green_y)+torch.square(covariant_derivative_blue_y)) #[HxW]
		g12 = self.beta*(torch.mul(covariant_derivative_red_x,covariant_derivative_red_y)+torch.mul(covariant_derivative_green_x,covariant_derivative_green_y)+torch.mul(covariant_derivative_blue_x,covariant_derivative_blue_y)) #[HxW]
		detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[HxW]


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_red_x)+torch.square(covariant_derivative_green_x)+torch.square(covariant_derivative_blue_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_red_x,covariant_derivative_red_y)+torch.mul(covariant_derivative_green_x,covariant_derivative_green_y)+torch.mul(covariant_derivative_blue_x,covariant_derivative_blue_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_red_y)+torch.square(covariant_derivative_green_y)+torch.square(covariant_derivative_blue_y))


		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_red_xx)+torch.square(covariant_derivative_green_xx)+torch.square(covariant_derivative_blue_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_red_xy)+torch.square(covariant_derivative_green_xy)+torch.square(covariant_derivative_blue_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_red_yy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_yy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_yy,covariant_derivative_blue_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_red_yy)+torch.square(covariant_derivative_green_yy)+torch.square(covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_yy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_yy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xy,covariant_derivative_blue_xy))
						

		norm_regularizer3_squared = torch.mul(torch.pow(invg11,3),torch.square(covariant_derivative_red_xxx)+torch.square(covariant_derivative_green_xxx)+torch.square(covariant_derivative_blue_xxx)) \
									+ torch.mul(torch.pow(invg22,3),torch.square(covariant_derivative_red_yyy)+torch.square(covariant_derivative_green_yyy)+torch.square(covariant_derivative_blue_yyy)) \
									+2.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_red_xxx,covariant_derivative_red_yyy)+torch.mul(covariant_derivative_green_xxx,covariant_derivative_green_yyy)+torch.mul(covariant_derivative_blue_xxx,covariant_derivative_blue_yyy)) \
									+6.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_red_xxy,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_xxy,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_xxy,covariant_derivative_blue_xyy)) \
									+12.*torch.mul(torch.mul(torch.mul(invg11,invg22),invg12),torch.mul(covariant_derivative_red_xxy,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_xxy,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_xxy,covariant_derivative_blue_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg11),invg12),torch.mul(covariant_derivative_red_xxx,covariant_derivative_red_xxy)+torch.mul(covariant_derivative_green_xxx,covariant_derivative_green_xxy)+torch.mul(covariant_derivative_blue_xxx,covariant_derivative_blue_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg22),invg12),torch.mul(covariant_derivative_red_yyy,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_yyy,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_yyy,covariant_derivative_blue_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.mul(covariant_derivative_red_xxx,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_xxx,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_xxx,covariant_derivative_blue_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.square(covariant_derivative_red_xxy)+torch.square(covariant_derivative_green_xxy)+torch.square(covariant_derivative_blue_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.mul(covariant_derivative_red_yyy,covariant_derivative_red_xxy)+torch.mul(covariant_derivative_green_yyy,covariant_derivative_green_xxy)+torch.mul(covariant_derivative_blue_yyy,covariant_derivative_blue_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.square(covariant_derivative_red_xyy)+torch.square(covariant_derivative_green_xyy)+torch.square(covariant_derivative_blue_xyy)) \
									+3.*torch.mul(torch.mul(torch.square(invg11),invg22),torch.square(covariant_derivative_red_xxy)+torch.square(covariant_derivative_green_xxy)+torch.square(covariant_derivative_blue_xxy)) \
									+3.*torch.mul(torch.mul(torch.square(invg22),invg11),torch.square(covariant_derivative_red_xyy)+torch.square(covariant_derivative_green_xyy)+torch.square(covariant_derivative_blue_xyy))

		norm_regularizer1 = torch.sqrt(epsilon + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]

		norm_regularizer2 = torch.sqrt(epsilon + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]

		norm_regularizer3 = torch.sqrt(epsilon + norm_regularizer3_squared) #[HxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		

		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 


class VectorBundleTotalVariationRplusRplusRplus3rdorderDeblurring2(nn.Module):
	def __init__(self,input_depth, pad, red, green, blue, diff_red_x,diff_green_x,diff_blue_x, diff_red_y,diff_green_y,diff_blue_y,diff_red_xx,diff_red_xy,diff_red_yy,diff_green_xx,diff_green_xy,diff_green_yy,diff_blue_xx,diff_blue_xy,diff_blue_yy, diff_red_xxx, diff_red_xxy, diff_red_xyy, diff_red_yyy, diff_green_xxx, diff_green_xxy, diff_green_xyy, diff_green_yyy, diff_blue_xxx, diff_blue_xxy, diff_blue_xyy, diff_blue_yyy, beta, epsilon, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorBundleTotalVariationRplusRplusRplus3rdorderDeblurring2,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon

		self.red0 = red
		self.green0 = green
		self.blue0 = blue

		self.diff_red0_x = diff_red_x
		self.diff_green0_x = diff_green_x
		self.diff_blue0_x = diff_blue_x
		self.diff_red0_y = diff_red_y
		self.diff_green0_y = diff_green_y
		self.diff_blue0_y = diff_blue_y

		self.diff_red0_xx = diff_red_xx
		self.diff_red0_xy = diff_red_xy
		self.diff_red0_yy = diff_red_yy

		self.diff_green0_xx = diff_green_xx
		self.diff_green0_xy = diff_green_xy
		self.diff_green0_yy = diff_green_yy

		self.diff_blue0_xx = diff_blue_xx
		self.diff_blue0_xy = diff_blue_xy
		self.diff_blue0_yy = diff_blue_yy

		self.diff_red0_xxx = diff_red_xxx
		self.diff_red0_xxy = diff_red_xxy
		self.diff_red0_xyy = diff_red_xyy
		self.diff_red0_yyy = diff_red_yyy

		self.diff_green0_xxx = diff_green_xxx
		self.diff_green0_xxy = diff_green_xxy
		self.diff_green0_xyy = diff_green_xyy
		self.diff_green0_yyy = diff_green_yyy

		self.diff_blue0_xxx = diff_blue_xxx
		self.diff_blue0_xxy = diff_blue_xxy
		self.diff_blue0_xyy = diff_blue_xyy
		self.diff_blue0_yyy = diff_blue_yyy


	def forward(self, input):	

		output = self.net(input)

		red = torch.narrow(output,1,0,1) #[1x1xHxW] 
		red = torch.squeeze(red) #[HxW]

		green = torch.narrow(output,1,1,1) #[1x1xHxW] 
		green = torch.squeeze(green) #[HxW]

		blue = torch.narrow(output,1,2,1) #[1x1xHxW] 
		blue = torch.squeeze(blue) #[HxW]

		diff_red_x = derivative_central_x_greylevel(red,dtype)     #[HxW]
		diff_green_x = derivative_central_x_greylevel(green,dtype) #[HxW]
		diff_blue_x = derivative_central_x_greylevel(blue,dtype)   #[HxW]

		diff_red_y = derivative_central_y_greylevel(red,dtype)     #[HxW]
		diff_green_y = derivative_central_y_greylevel(green,dtype) #[HxW]
		diff_blue_y = derivative_central_y_greylevel(blue,dtype)   #[HxW]

		diff_red_xx = derivative_xx_greylevel(red,dtype) #[HxW]
		diff_red_xy = derivative_xy_greylevel(red,dtype) #[HxW]
		diff_red_yy = derivative_yy_greylevel(red,dtype) #[HxW]

		diff_green_xx = derivative_xx_greylevel(green,dtype) #[HxW]
		diff_green_xy = derivative_xy_greylevel(green,dtype) #[HxW]
		diff_green_yy = derivative_yy_greylevel(green,dtype) #[HxW]

		diff_blue_xx = derivative_xx_greylevel(blue,dtype) #[HxW]
		diff_blue_xy = derivative_xy_greylevel(blue,dtype) #[HxW]
		diff_blue_yy = derivative_yy_greylevel(blue,dtype) #[HxW]

		diff_red_xxx = derivative_xxx_greylevel(red,dtype) #[HxW]
		diff_red_xxy = derivative_xxy_greylevel(red,dtype) #[HxW]
		diff_red_xyy = derivative_xyy_greylevel(red,dtype) #[HxW]
		diff_red_yyy = derivative_yyy_greylevel(red,dtype) #[HxW]

		diff_green_xxx = derivative_xxx_greylevel(green,dtype) #[HxW]
		diff_green_xxy = derivative_xxy_greylevel(green,dtype) #[HxW]
		diff_green_xyy = derivative_xyy_greylevel(green,dtype) #[HxW]
		diff_green_yyy = derivative_yyy_greylevel(green,dtype) #[HxW]

		diff_blue_xxx = derivative_xxx_greylevel(blue,dtype) #[HxW]
		diff_blue_xxy = derivative_xxy_greylevel(blue,dtype) #[HxW]
		diff_blue_xyy = derivative_xyy_greylevel(blue,dtype) #[HxW]
		diff_blue_yyy = derivative_yyy_greylevel(blue,dtype) #[HxW]


		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		covariant_derivative_red_x = diff_red_x + torch.mul(torch.div(self.diff_red0_x,epsilon+self.red0),red)
		covariant_derivative_red_y = diff_red_y + torch.mul(torch.div(self.diff_red0_y,epsilon+self.red0),red)

		covariant_derivative_green_x = diff_green_x + torch.mul(torch.div(self.diff_green0_x,epsilon+self.green0),green)
		covariant_derivative_green_y = diff_green_y + torch.mul(torch.div(self.diff_green0_y,epsilon+self.green0),green)

		covariant_derivative_blue_x = diff_blue_x + torch.mul(torch.div(self.diff_blue0_x,epsilon+self.blue0),blue)
		covariant_derivative_blue_y = diff_blue_y + torch.mul(torch.div(self.diff_blue0_y,epsilon+self.blue0),blue)


		covariant_derivative_red_xx = diff_red_xx + torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xx) + torch.mul(torch.mul(torch.div(2.,epsilon+self.red0),self.diff_red0_x),diff_red_x)
		covariant_derivative_red_xy = diff_red_xy + torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xy) + torch.mul(torch.div(1.,epsilon+self.red0), torch.mul(diff_red_x,self.diff_red0_y)+ torch.mul(diff_red_y,self.diff_red0_x)) 
		covariant_derivative_red_yy = diff_red_yy + torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_yy) + torch.mul(torch.mul(torch.div(2.,epsilon+self.red0),self.diff_red0_y),diff_red_y)
					

		covariant_derivative_green_xx = diff_green_xx + torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xx) + torch.mul(torch.mul(torch.div(2.,epsilon+self.green0),self.diff_green0_x),diff_green_x)
		covariant_derivative_green_xy = diff_green_xy + torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xy) + torch.mul(torch.div(1.,epsilon+self.green0), torch.mul(diff_green_x,self.diff_green0_y)+ torch.mul(diff_green_y,self.diff_green0_x)) 
		covariant_derivative_green_yy = diff_green_yy + torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_yy) + torch.mul(torch.mul(torch.div(2.,epsilon+self.green0),self.diff_green0_y),diff_green_y)
		
		covariant_derivative_blue_xx = diff_blue_xx + torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xx) + torch.mul(torch.mul(torch.div(2.,epsilon+self.blue0),self.diff_blue0_x),diff_blue_x)
		covariant_derivative_blue_xy = diff_blue_xy + torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xy) + torch.mul(torch.div(1.,epsilon+self.blue0), torch.mul(diff_blue_x,self.diff_blue0_y)+ torch.mul(diff_blue_y,self.diff_blue0_x)) 
		covariant_derivative_blue_yy = diff_blue_yy + torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_yy) + torch.mul(torch.mul(torch.div(2.,epsilon+self.blue0),self.diff_blue0_y),diff_blue_y)

		covariant_derivative_red_xxx = diff_red_xxx \
									   + torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xxx) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.red0), torch.mul(self.diff_red0_x,diff_red_xx)) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.red0),torch.mul(self.diff_red0_xx,diff_red_x))
									   

		covariant_derivative_red_xxy = diff_red_xxy \
		                               + torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xxy) \
		                               + torch.mul(torch.div(1.,epsilon+self.red0),2.*torch.mul(self.diff_red0_x,diff_red_xy)+torch.mul(self.diff_red0_y,diff_red_xx)) \
									   + torch.mul(torch.div(1.,epsilon+self.red0),2.*torch.mul(self.diff_red0_xy,diff_red_x)+torch.mul(self.diff_red0_xx,diff_red_y))
									   

		covariant_derivative_red_xyy = diff_red_xyy \
									   + torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xyy) \
									   + torch.mul(torch.div(1.,epsilon+self.red0),2.*torch.mul(self.diff_red0_y,diff_red_xy)+torch.mul(self.diff_red0_x,diff_red_yy)) \
									   + torch.mul(torch.div(1.,epsilon+self.red0),2.*torch.mul(self.diff_red0_xy,diff_red_y)+torch.mul(self.diff_red0_yy,diff_red_x))
									   
		covariant_derivative_red_yyy = diff_red_yyy \
									   + torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_yyy) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.red0), torch.mul(self.diff_red0_y,diff_red_yy)) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.red0),torch.mul(self.diff_red0_yy,diff_red_y)) 
									  
	
		covariant_derivative_green_xxx = diff_green_xxx \
									   + torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xxx) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.green0), torch.mul(self.diff_green0_x,diff_green_xx)) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.green0),torch.mul(self.diff_green0_xx,diff_green_x))
									   

		covariant_derivative_green_xxy = diff_green_xxy \
		                               + torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xxy) \
		                               + torch.mul(torch.div(1.,epsilon+self.green0),2.*torch.mul(self.diff_green0_x,diff_green_xy)+torch.mul(self.diff_green0_y,diff_green_xx)) \
									   + torch.mul(torch.div(1.,epsilon+self.green0),2.*torch.mul(self.diff_green0_xy,diff_green_x)+torch.mul(self.diff_green0_xx,diff_green_y))
									   

		covariant_derivative_green_xyy = diff_green_xyy \
									   + torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xyy) \
									   + torch.mul(torch.div(1.,epsilon+self.green0),2.*torch.mul(self.diff_green0_y,diff_green_xy)+torch.mul(self.diff_green0_x,diff_green_yy)) \
									   + torch.mul(torch.div(1.,epsilon+self.green0),2.*torch.mul(self.diff_green0_xy,diff_green_y)+torch.mul(self.diff_green0_yy,diff_green_x))
									   
		covariant_derivative_green_yyy = diff_green_yyy \
									   + torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_yyy) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.green0), torch.mul(self.diff_green0_y,diff_green_yy)) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.green0),torch.mul(self.diff_green0_yy,diff_green_y))
									   


		covariant_derivative_blue_xxx = diff_blue_xxx \
									   + torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xxx) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.blue0), torch.mul(self.diff_blue0_x,diff_blue_xx)) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.blue0),torch.mul(self.diff_blue0_xx,diff_blue_x))
									   

		covariant_derivative_blue_xxy = diff_blue_xxy \
		                               + torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xxy) \
		                               + torch.mul(torch.div(1.,epsilon+self.blue0),2.*torch.mul(self.diff_blue0_x,diff_blue_xy)+torch.mul(self.diff_blue0_y,diff_blue_xx)) \
									   + torch.mul(torch.div(1.,epsilon+self.blue0),2.*torch.mul(self.diff_blue0_xy,diff_blue_x)+torch.mul(self.diff_blue0_xx,diff_blue_y))
									   

		covariant_derivative_blue_xyy = diff_blue_xyy \
									   + torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xyy) \
									   + torch.mul(torch.div(1.,epsilon+self.blue0),2.*torch.mul(self.diff_blue0_y,diff_blue_xy)+torch.mul(self.diff_blue0_x,diff_blue_yy)) \
									   + torch.mul(torch.div(1.,epsilon+self.blue0),2.*torch.mul(self.diff_blue0_xy,diff_blue_y)+torch.mul(self.diff_blue0_yy,diff_blue_x))
									   
		covariant_derivative_blue_yyy = diff_blue_yyy \
									   + torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_yyy) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.blue0), torch.mul(self.diff_blue0_y,diff_blue_yy)) \
									   + 3.*torch.mul(torch.div(1.,epsilon+self.blue0),torch.mul(self.diff_blue0_yy,diff_blue_y))

		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_red_x)+torch.square(covariant_derivative_green_x)+torch.square(covariant_derivative_blue_x)) #[HxW]
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_red_y)+torch.square(covariant_derivative_green_y)+torch.square(covariant_derivative_blue_y)) #[HxW]
		g12 = self.beta*(torch.mul(covariant_derivative_red_x,covariant_derivative_red_y)+torch.mul(covariant_derivative_green_x,covariant_derivative_green_y)+torch.mul(covariant_derivative_blue_x,covariant_derivative_blue_y)) #[HxW]
		detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[HxW]


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_red_x)+torch.square(covariant_derivative_green_x)+torch.square(covariant_derivative_blue_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_red_x,covariant_derivative_red_y)+torch.mul(covariant_derivative_green_x,covariant_derivative_green_y)+torch.mul(covariant_derivative_blue_x,covariant_derivative_blue_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_red_y)+torch.square(covariant_derivative_green_y)+torch.square(covariant_derivative_blue_y))


		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_red_xx)+torch.square(covariant_derivative_green_xx)+torch.square(covariant_derivative_blue_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_red_xy)+torch.square(covariant_derivative_green_xy)+torch.square(covariant_derivative_blue_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_red_yy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_yy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_yy,covariant_derivative_blue_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_red_yy)+torch.square(covariant_derivative_green_yy)+torch.square(covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_yy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_yy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xy,covariant_derivative_blue_xy))
						

		norm_regularizer3_squared = torch.mul(torch.pow(invg11,3),torch.square(covariant_derivative_red_xxx)+torch.square(covariant_derivative_green_xxx)+torch.square(covariant_derivative_blue_xxx)) \
									+ torch.mul(torch.pow(invg22,3),torch.square(covariant_derivative_red_yyy)+torch.square(covariant_derivative_green_yyy)+torch.square(covariant_derivative_blue_yyy)) \
									+2.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_red_xxx,covariant_derivative_red_yyy)+torch.mul(covariant_derivative_green_xxx,covariant_derivative_green_yyy)+torch.mul(covariant_derivative_blue_xxx,covariant_derivative_blue_yyy)) \
									+6.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_red_xxy,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_xxy,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_xxy,covariant_derivative_blue_xyy)) \
									+12.*torch.mul(torch.mul(torch.mul(invg11,invg22),invg12),torch.mul(covariant_derivative_red_xxy,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_xxy,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_xxy,covariant_derivative_blue_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg11),invg12),torch.mul(covariant_derivative_red_xxx,covariant_derivative_red_xxy)+torch.mul(covariant_derivative_green_xxx,covariant_derivative_green_xxy)+torch.mul(covariant_derivative_blue_xxx,covariant_derivative_blue_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg22),invg12),torch.mul(covariant_derivative_red_yyy,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_yyy,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_yyy,covariant_derivative_blue_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.mul(covariant_derivative_red_xxx,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_xxx,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_xxx,covariant_derivative_blue_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.square(covariant_derivative_red_xxy)+torch.square(covariant_derivative_green_xxy)+torch.square(covariant_derivative_blue_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.mul(covariant_derivative_red_yyy,covariant_derivative_red_xxy)+torch.mul(covariant_derivative_green_yyy,covariant_derivative_green_xxy)+torch.mul(covariant_derivative_blue_yyy,covariant_derivative_blue_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.square(covariant_derivative_red_xyy)+torch.square(covariant_derivative_green_xyy)+torch.square(covariant_derivative_blue_xyy)) \
									+3.*torch.mul(torch.mul(torch.square(invg11),invg22),torch.square(covariant_derivative_red_xxy)+torch.square(covariant_derivative_green_xxy)+torch.square(covariant_derivative_blue_xxy)) \
									+3.*torch.mul(torch.mul(torch.square(invg22),invg11),torch.square(covariant_derivative_red_xyy)+torch.square(covariant_derivative_green_xyy)+torch.square(covariant_derivative_blue_xyy))

		norm_regularizer1 = torch.sqrt(epsilon + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]

		norm_regularizer2 = torch.sqrt(epsilon + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]

		norm_regularizer3 = torch.sqrt(epsilon + norm_regularizer3_squared) #[HxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		

		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 



class RiemannianVectorialTotalVariation3rdorderDeblurring(nn.Module):
	def __init__(self,input_depth, pad, beta, epsilon, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(RiemannianVectorialTotalVariation3rdorderDeblurring,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon


	def forward(self, input):	

		output = self.net(input)

		red = torch.narrow(output,1,0,1) #[1x1xHxW] 
		red = torch.squeeze(red) #[HxW]

		green = torch.narrow(output,1,1,1) #[1x1xHxW] 
		green = torch.squeeze(green) #[HxW]

		blue = torch.narrow(output,1,2,1) #[1x1xHxW] 
		blue = torch.squeeze(blue) #[HxW]

		covariant_derivative_red_x = derivative_central_x_greylevel(red,dtype)     #[HxW]
		covariant_derivative_green_x = derivative_central_x_greylevel(green,dtype) #[HxW]
		covariant_derivative_blue_x = derivative_central_x_greylevel(blue,dtype)   #[HxW]

		covariant_derivative_red_y = derivative_central_y_greylevel(red,dtype)     #[HxW]
		covariant_derivative_green_y = derivative_central_y_greylevel(green,dtype) #[HxW]
		covariant_derivative_blue_y = derivative_central_y_greylevel(blue,dtype)   #[HxW]

		covariant_derivative_red_xx = derivative_xx_greylevel(red,dtype) #[HxW]
		covariant_derivative_red_xy = derivative_xy_greylevel(red,dtype) #[HxW]
		covariant_derivative_red_yy = derivative_yy_greylevel(red,dtype) #[HxW]

		covariant_derivative_green_xx = derivative_xx_greylevel(green,dtype) #[HxW]
		covariant_derivative_green_xy = derivative_xy_greylevel(green,dtype) #[HxW]
		covariant_derivative_green_yy = derivative_yy_greylevel(green,dtype) #[HxW]

		covariant_derivative_blue_xx = derivative_xx_greylevel(blue,dtype) #[HxW]
		covariant_derivative_blue_xy = derivative_xy_greylevel(blue,dtype) #[HxW]
		covariant_derivative_blue_yy = derivative_yy_greylevel(blue,dtype) #[HxW]

		covariant_derivative_red_xxx = derivative_xxx_greylevel(red,dtype) #[HxW]
		covariant_derivative_red_xxy = derivative_xxy_greylevel(red,dtype) #[HxW]
		covariant_derivative_red_xyy = derivative_xyy_greylevel(red,dtype) #[HxW]
		covariant_derivative_red_yyy = derivative_yyy_greylevel(red,dtype) #[HxW]

		covariant_derivative_green_xxx = derivative_xxx_greylevel(green,dtype) #[HxW]
		covariant_derivative_green_xxy = derivative_xxy_greylevel(green,dtype) #[HxW]
		covariant_derivative_green_xyy = derivative_xyy_greylevel(green,dtype) #[HxW]
		covariant_derivative_green_yyy = derivative_yyy_greylevel(green,dtype) #[HxW]

		covariant_derivative_blue_xxx = derivative_xxx_greylevel(blue,dtype) #[HxW]
		covariant_derivative_blue_xxy = derivative_xxy_greylevel(blue,dtype) #[HxW]
		covariant_derivative_blue_xyy = derivative_xyy_greylevel(blue,dtype) #[HxW]
		covariant_derivative_blue_yyy = derivative_yyy_greylevel(blue,dtype) #[HxW]


		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_red_x)+torch.square(covariant_derivative_green_x)+torch.square(covariant_derivative_blue_x)) #[HxW]
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_red_y)+torch.square(covariant_derivative_green_y)+torch.square(covariant_derivative_blue_y)) #[HxW]
		g12 = self.beta*(torch.mul(covariant_derivative_red_x,covariant_derivative_red_y)+torch.mul(covariant_derivative_green_x,covariant_derivative_green_y)+torch.mul(covariant_derivative_blue_x,covariant_derivative_blue_y)) #[HxW]
		detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[HxW]


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_red_x)+torch.square(covariant_derivative_green_x)+torch.square(covariant_derivative_blue_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_red_x,covariant_derivative_red_y)+torch.mul(covariant_derivative_green_x,covariant_derivative_green_y)+torch.mul(covariant_derivative_blue_x,covariant_derivative_blue_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_red_y)+torch.square(covariant_derivative_green_y)+torch.square(covariant_derivative_blue_y))


		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_red_xx)+torch.square(covariant_derivative_green_xx)+torch.square(covariant_derivative_blue_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_red_xy)+torch.square(covariant_derivative_green_xy)+torch.square(covariant_derivative_blue_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_red_yy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_yy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_yy,covariant_derivative_blue_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_red_yy)+torch.square(covariant_derivative_green_yy)+torch.square(covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_yy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_yy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xy,covariant_derivative_blue_xy))
						

		norm_regularizer3_squared = torch.mul(torch.pow(invg11,3),torch.square(covariant_derivative_red_xxx)+torch.square(covariant_derivative_green_xxx)+torch.square(covariant_derivative_blue_xxx)) \
									+ torch.mul(torch.pow(invg22,3),torch.square(covariant_derivative_red_yyy)+torch.square(covariant_derivative_green_yyy)+torch.square(covariant_derivative_blue_yyy)) \
									+2.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_red_xxx,covariant_derivative_red_yyy)+torch.mul(covariant_derivative_green_xxx,covariant_derivative_green_yyy)+torch.mul(covariant_derivative_blue_xxx,covariant_derivative_blue_yyy)) \
									+6.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_red_xxy,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_xxy,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_xxy,covariant_derivative_blue_xyy)) \
									+12.*torch.mul(torch.mul(torch.mul(invg11,invg22),invg12),torch.mul(covariant_derivative_red_xxy,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_xxy,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_xxy,covariant_derivative_blue_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg11),invg12),torch.mul(covariant_derivative_red_xxx,covariant_derivative_red_xxy)+torch.mul(covariant_derivative_green_xxx,covariant_derivative_green_xxy)+torch.mul(covariant_derivative_blue_xxx,covariant_derivative_blue_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg22),invg12),torch.mul(covariant_derivative_red_yyy,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_yyy,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_yyy,covariant_derivative_blue_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.mul(covariant_derivative_red_xxx,covariant_derivative_red_xyy)+torch.mul(covariant_derivative_green_xxx,covariant_derivative_green_xyy)+torch.mul(covariant_derivative_blue_xxx,covariant_derivative_blue_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.square(covariant_derivative_red_xxy)+torch.square(covariant_derivative_green_xxy)+torch.square(covariant_derivative_blue_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.mul(covariant_derivative_red_yyy,covariant_derivative_red_xxy)+torch.mul(covariant_derivative_green_yyy,covariant_derivative_green_xxy)+torch.mul(covariant_derivative_blue_yyy,covariant_derivative_blue_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.square(covariant_derivative_red_xyy)+torch.square(covariant_derivative_green_xyy)+torch.square(covariant_derivative_blue_xyy)) \
									+3.*torch.mul(torch.mul(torch.square(invg11),invg22),torch.square(covariant_derivative_red_xxy)+torch.square(covariant_derivative_green_xxy)+torch.square(covariant_derivative_blue_xxy)) \
									+3.*torch.mul(torch.mul(torch.square(invg22),invg11),torch.square(covariant_derivative_red_xyy)+torch.square(covariant_derivative_green_xyy)+torch.square(covariant_derivative_blue_xyy))

		norm_regularizer1 = torch.sqrt(epsilon + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]

		norm_regularizer2 = torch.sqrt(epsilon + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]

		norm_regularizer3 = torch.sqrt(epsilon + norm_regularizer3_squared) #[HxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		

		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 


class RiemannianVectorialTotalVariation3rdorderOpponentSpace(nn.Module):
	def __init__(self,input_depth, pad, beta, epsilon, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(RiemannianVectorialTotalVariation3rdorderOpponentSpace,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon


	def forward(self, input):	

		output = self.net(input)

		luminance = (1/(sqrt(3)))*torch.sum(output,dim=1)
		lum = torch.squeeze(luminance)/0.6 #[HxW]

		chrominance1 = (1/sqrt(2))*torch.narrow(output,1,0,1) - (1/sqrt(2))*torch.narrow(output,1,1,1)
		chrom1 = torch.squeeze(chrominance1) #[HxW]
		chrominance2 = (1/sqrt(6))*torch.narrow(output,1,0,1) + (1/sqrt(6))*torch.narrow(output,1,1,1) - (2/sqrt(6))*torch.narrow(output,1,2,1)
		chrom2 = torch.squeeze(chrominance2) #[HxW]

		

		covariant_derivative_lum_x = derivative_central_x_greylevel(lum,dtype)    #[HxW]
		covariant_derivative_chrom1_x = derivative_central_x_greylevel(chrom1,dtype) #[HxW]
		covariant_derivative_chrom2_x = derivative_central_x_greylevel(chrom2,dtype)   #[HxW]

		covariant_derivative_lum_y = derivative_central_y_greylevel(lum,dtype)     #[HxW]
		covariant_derivative_chrom1_y = derivative_central_y_greylevel(chrom1,dtype) #[HxW]
		covariant_derivative_chrom2_y = derivative_central_y_greylevel(chrom2,dtype)   #[HxW]

		covariant_derivative_lum_xx = derivative_xx_greylevel(lum,dtype) #[HxW]
		covariant_derivative_lum_xy = derivative_xy_greylevel(lum,dtype) #[HxW]
		covariant_derivative_lum_yy = derivative_yy_greylevel(lum,dtype) #[HxW]

		covariant_derivative_chrom1_xx = derivative_xx_greylevel(chrom1,dtype) #[HxW]
		covariant_derivative_chrom1_xy = derivative_xy_greylevel(chrom1,dtype) #[HxW]
		covariant_derivative_chrom1_yy = derivative_yy_greylevel(chrom1,dtype) #[HxW]

		covariant_derivative_chrom2_xx = derivative_xx_greylevel(chrom2,dtype) #[HxW]
		covariant_derivative_chrom2_xy = derivative_xy_greylevel(chrom2,dtype) #[HxW]
		covariant_derivative_chrom2_yy = derivative_yy_greylevel(chrom2,dtype) #[HxW]

		covariant_derivative_lum_xxx = derivative_xxx_greylevel(lum,dtype) #[HxW]
		covariant_derivative_lum_xxy = derivative_xxy_greylevel(lum,dtype) #[HxW]
		covariant_derivative_lum_xyy = derivative_xyy_greylevel(lum,dtype) #[HxW]
		covariant_derivative_lum_yyy = derivative_yyy_greylevel(lum,dtype) #[HxW]

		covariant_derivative_chrom1_xxx = derivative_xxx_greylevel(chrom1,dtype) #[HxW]
		covariant_derivative_chrom1_xxy = derivative_xxy_greylevel(chrom1,dtype) #[HxW]
		covariant_derivative_chrom1_xyy = derivative_xyy_greylevel(chrom1,dtype) #[HxW]
		covariant_derivative_chrom1_yyy = derivative_yyy_greylevel(chrom1,dtype) #[HxW]

		covariant_derivative_chrom2_xxx = derivative_xxx_greylevel(chrom2,dtype) #[HxW]
		covariant_derivative_chrom2_xxy = derivative_xxy_greylevel(chrom2,dtype) #[HxW]
		covariant_derivative_chrom2_xyy = derivative_xyy_greylevel(chrom2,dtype) #[HxW]
		covariant_derivative_chrom2_yyy = derivative_yyy_greylevel(chrom2,dtype) #[HxW]


		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_lum_x)+torch.square(covariant_derivative_chrom1_x)+torch.square(covariant_derivative_chrom2_x)) #[HxW]
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_lum_y)+torch.square(covariant_derivative_chrom1_y)+torch.square(covariant_derivative_chrom2_y)) #[HxW]
		g12 = self.beta*(torch.mul(covariant_derivative_lum_x,covariant_derivative_lum_y)+torch.mul(covariant_derivative_chrom1_x,covariant_derivative_chrom1_y)+torch.mul(covariant_derivative_chrom2_x,covariant_derivative_chrom2_y)) #[HxW]
		detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[HxW]


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_lum_x)+torch.square(covariant_derivative_chrom1_x)+torch.square(covariant_derivative_chrom2_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_lum_x,covariant_derivative_lum_y)+torch.mul(covariant_derivative_chrom1_x,covariant_derivative_chrom1_y)+torch.mul(covariant_derivative_chrom2_x,covariant_derivative_chrom2_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_lum_y)+torch.square(covariant_derivative_chrom1_y)+torch.square(covariant_derivative_chrom2_y))


		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_lum_xx)+torch.square(covariant_derivative_chrom1_xx)+torch.square(covariant_derivative_chrom2_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_lum_xx,covariant_derivative_lum_xy)+torch.mul(covariant_derivative_chrom1_xx,covariant_derivative_chrom1_xy)+torch.mul(covariant_derivative_chrom2_xx,covariant_derivative_chrom2_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_lum_xy)+torch.square(covariant_derivative_chrom1_xy)+torch.square(covariant_derivative_chrom2_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_lum_yy,covariant_derivative_lum_xy)+torch.mul(covariant_derivative_chrom1_yy,covariant_derivative_chrom1_xy)+torch.mul(covariant_derivative_chrom2_yy,covariant_derivative_chrom2_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_lum_yy)+torch.square(covariant_derivative_chrom1_yy)+torch.square(covariant_derivative_chrom2_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_lum_xx,covariant_derivative_lum_yy)+torch.mul(covariant_derivative_chrom1_xx,covariant_derivative_chrom1_yy)+torch.mul(covariant_derivative_chrom2_xx,covariant_derivative_chrom2_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_lum_xy,covariant_derivative_lum_xy)+torch.mul(covariant_derivative_chrom1_xy,covariant_derivative_chrom1_xy)+torch.mul(covariant_derivative_chrom2_xy,covariant_derivative_chrom2_xy))
						

		norm_regularizer3_squared = torch.mul(torch.pow(invg11,3),torch.square(covariant_derivative_lum_xxx)+torch.square(covariant_derivative_chrom1_xxx)+torch.square(covariant_derivative_chrom2_xxx)) \
									+ torch.mul(torch.pow(invg22,3),torch.square(covariant_derivative_lum_yyy)+torch.square(covariant_derivative_chrom1_yyy)+torch.square(covariant_derivative_chrom2_yyy)) \
									+2.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_lum_xxx,covariant_derivative_lum_yyy)+torch.mul(covariant_derivative_chrom1_xxx,covariant_derivative_chrom1_yyy)+torch.mul(covariant_derivative_chrom2_xxx,covariant_derivative_chrom2_yyy)) \
									+6.*torch.mul(torch.pow(invg12,3),torch.mul(covariant_derivative_lum_xxy,covariant_derivative_lum_xyy)+torch.mul(covariant_derivative_chrom1_xxy,covariant_derivative_chrom1_xyy)+torch.mul(covariant_derivative_chrom2_xxy,covariant_derivative_chrom2_xyy)) \
									+12.*torch.mul(torch.mul(torch.mul(invg11,invg22),invg12),torch.mul(covariant_derivative_lum_xxy,covariant_derivative_lum_xyy)+torch.mul(covariant_derivative_chrom1_xxy,covariant_derivative_chrom1_xyy)+torch.mul(covariant_derivative_chrom2_xxy,covariant_derivative_chrom2_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg11),invg12),torch.mul(covariant_derivative_lum_xxx,covariant_derivative_lum_xxy)+torch.mul(covariant_derivative_chrom1_xxx,covariant_derivative_chrom1_xxy)+torch.mul(covariant_derivative_chrom2_xxx,covariant_derivative_chrom2_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg22),invg12),torch.mul(covariant_derivative_lum_yyy,covariant_derivative_lum_xyy)+torch.mul(covariant_derivative_chrom1_yyy,covariant_derivative_chrom1_xyy)+torch.mul(covariant_derivative_chrom2_yyy,covariant_derivative_chrom2_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.mul(covariant_derivative_lum_xxx,covariant_derivative_lum_xyy)+torch.mul(covariant_derivative_chrom1_xxx,covariant_derivative_chrom1_xyy)+torch.mul(covariant_derivative_chrom2_xxx,covariant_derivative_chrom2_xyy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg11),torch.square(covariant_derivative_lum_xxy)+torch.square(covariant_derivative_chrom1_xxy)+torch.square(covariant_derivative_chrom2_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.mul(covariant_derivative_lum_yyy,covariant_derivative_lum_xxy)+torch.mul(covariant_derivative_chrom1_yyy,covariant_derivative_chrom1_xxy)+torch.mul(covariant_derivative_chrom2_yyy,covariant_derivative_chrom2_xxy)) \
									+6.*torch.mul(torch.mul(torch.square(invg12),invg22),torch.square(covariant_derivative_lum_xyy)+torch.square(covariant_derivative_chrom1_xyy)+torch.square(covariant_derivative_chrom2_xyy)) \
									+3.*torch.mul(torch.mul(torch.square(invg11),invg22),torch.square(covariant_derivative_lum_xxy)+torch.square(covariant_derivative_chrom1_xxy)+torch.square(covariant_derivative_chrom2_xxy)) \
									+3.*torch.mul(torch.mul(torch.square(invg22),invg11),torch.square(covariant_derivative_lum_xyy)+torch.square(covariant_derivative_chrom1_xyy)+torch.square(covariant_derivative_chrom2_xyy))

		norm_regularizer1 = torch.sqrt(epsilon + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]

		norm_regularizer2 = torch.sqrt(epsilon + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]

		norm_regularizer3 = torch.sqrt(epsilon + norm_regularizer3_squared) #[HxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		

		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 


class RiemannianVectorialTotalVariation3rdorderDeblurringOpponentSpace2(nn.Module):
	def __init__(self,input_depth, pad, beta, epsilon, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(RiemannianVectorialTotalVariation3rdorderDeblurringOpponentSpace2,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon

	def forward(self, input):	

		output = self.net(input) 

		img = torch.squeeze(output) #[3xHxW]

		M = img.size(dim=1)
		N = img.size(dim=2)

		luminance = (1/(0.6*sqrt(3)))*torch.sum(img,dim=0) #[HxW]
		lum = torch.unsqueeze(luminance,dim=0) #[1xHxW]

		chrominance1 = (1/sqrt(2))*torch.narrow(img,0,0,1) - (1/sqrt(2))*torch.narrow(img,0,1,1)
		#chrom1 = torch.unsqueeze(chrominance1,dim=0) #[1xHxW]
		chrominance2 = (1/sqrt(6))*torch.narrow(img,0,0,1) + (1/sqrt(6))*torch.narrow(img,0,1,1) - (2/sqrt(6))*torch.narrow(img,0,2,1)
		#chrom2 = torch.unsqueeze(chrominance2,dim=0) #[1xHxW]

		img_opp = torch.cat((lum,chrominance1,chrominance2),dim=0) #[3xHxW]


		Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11 = Levi_Cevita_connection_coefficients_color_standard_frame(img_opp,self.beta,dtype)

		g11,g12,g22 = riemannianmetric_color(img_opp,self.beta,dtype)

		detg = torch.mul(g11,g22) - torch.mul(g12,g12) 
		invdetg = torch.div(torch.ones(M,N).type(dtype),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)

		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(dtype)

		norm_regularizer1 = norm_first_order_covariant_derivative_color(img_opp,epsilon,self.beta,dtype,invg11,invg12,invg22)							
		norm_regularizer2 = norm_second_order_covariant_derivative_color(img_opp,epsilon,self.beta,dtype,invg11,invg12,invg22,Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11)
		norm_regularizer3 = norm_third_order_covariant_derivative_color(img_opp,epsilon,self.beta,dtype,invg11,invg12,invg22,Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11)
						
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]
		norm_regularizer3 = torch.unsqueeze(norm_regularizer3,dim=0) #[1xHxW]
		
		return output, norm_regularizer1, norm_regularizer2, norm_regularizer3 




'''class VectorBundleTotalVariationRplusRplusRplus2ndorderDualDeblurring(nn.Module):
	def __init__(self,input_depth, pad, red, green, blue, diff_red_x,diff_green_x,diff_blue_x, diff_red_y,diff_green_y,diff_blue_y,diff_red_xx,diff_red_xy,diff_red_yy,diff_green_xx,diff_green_xy,diff_green_yy,diff_blue_xx,diff_blue_xy,diff_blue_yy, beta, epsilon, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorBundleTotalVariationRplusRplusRplus2ndorderDualDeblurring,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.epsilon = epsilon

		self.red0 = red
		self.green0 = green
		self.blue0 = blue

		self.diff_red0_x = diff_red_x
		self.diff_green0_x = diff_green_x
		self.diff_blue0_x = diff_blue_x
		self.diff_red0_y = diff_red_y
		self.diff_green0_y = diff_green_y
		self.diff_blue0_y = diff_blue_y

		self.diff_red0_xx = diff_red_xx
		self.diff_red0_xy = diff_red_xy
		self.diff_red0_yy = diff_red_yy

		self.diff_green0_xx = diff_green_xx
		self.diff_green0_xy = diff_green_xy
		self.diff_green0_yy = diff_green_yy

		self.diff_blue0_xx = diff_blue_xx
		self.diff_blue0_xy = diff_blue_xy
		self.diff_blue0_yy = diff_blue_yy


	def forward(self, input):	

		output = self.net(input)

		red = torch.narrow(output,1,0,1) #[1x1xHxW] 
		red = torch.squeeze(red) #[HxW]

		green = torch.narrow(output,1,1,1) #[1x1xHxW] 
		green = torch.squeeze(green) #[HxW]

		blue = torch.narrow(output,1,2,1) #[1x1xHxW] 
		blue = torch.squeeze(blue) #[HxW]

		diff_red_x = derivative_central_x_greylevel(red,dtype)     #[HxW]
		diff_green_x = derivative_central_x_greylevel(green,dtype) #[HxW]
		diff_blue_x = derivative_central_x_greylevel(blue,dtype)   #[HxW]

		diff_red_y = derivative_central_y_greylevel(red,dtype)     #[HxW]
		diff_green_y = derivative_central_y_greylevel(green,dtype) #[HxW]
		diff_blue_y = derivative_central_y_greylevel(blue,dtype)   #[HxW]

		diff_red_xx = derivative_xx_greylevel(red,dtype) #[HxW]
		diff_red_xy = derivative_xy_greylevel(red,dtype) #[HxW]
		diff_red_yy = derivative_yy_greylevel(red,dtype) #[HxW]

		diff_green_xx = derivative_xx_greylevel(green,dtype) #[HxW]
		diff_green_xy = derivative_xy_greylevel(green,dtype) #[HxW]
		diff_green_yy = derivative_yy_greylevel(green,dtype) #[HxW]

		diff_blue_xx = derivative_xx_greylevel(blue,dtype) #[HxW]
		diff_blue_xy = derivative_xy_greylevel(blue,dtype) #[HxW]
		diff_blue_yy = derivative_yy_greylevel(blue,dtype) #[HxW]

		epsilon = self.epsilon*torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor)

		covariant_derivative_red_x = diff_red_x + torch.mul(torch.div(self.diff_red0_x,epsilon+self.red0),red)
		covariant_derivative_red_y = diff_red_y + torch.mul(torch.div(self.diff_red0_y,epsilon+self.red0),red)

		covariant_derivative_green_x = diff_green_x + torch.mul(torch.div(self.diff_green0_x,epsilon+self.green0),green)
		covariant_derivative_green_y = diff_green_y + torch.mul(torch.div(self.diff_green0_y,epsilon+self.green0),green)

		covariant_derivative_blue_x = diff_blue_x + torch.mul(torch.div(self.diff_blue0_x,epsilon+self.blue0),blue)
		covariant_derivative_blue_y = diff_blue_y + torch.mul(torch.div(self.diff_blue0_y,epsilon+self.blue0),blue)


		covariant_derivative_red_xx = diff_red_xx + torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xx) + torch.mul(torch.mul(torch.div(2.,epsilon+self.red0),self.diff_red0_x),diff_red_x)
		covariant_derivative_red_xy = diff_red_xy + torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_xy) + torch.mul(torch.div(1.,epsilon+self.red0), torch.mul(diff_red_x,self.diff_red0_y)+ torch.mul(diff_red_y,self.diff_red0_x))
		covariant_derivative_red_yy = diff_red_yy + torch.mul(torch.div(red,epsilon+self.red0),self.diff_red0_yy) + torch.mul(torch.mul(torch.div(2.,epsilon+self.red0),self.diff_red0_y),diff_red_y)
					

		covariant_derivative_green_xx = diff_green_xx + torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xx) + torch.mul(torch.mul(torch.div(2.,epsilon+self.green0),self.diff_green0_x),diff_green_x)
		covariant_derivative_green_xy = diff_green_xy + torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_xy) + torch.mul(torch.div(1.,epsilon+self.green0), torch.mul(diff_green_x,self.diff_green0_y)+ torch.mul(diff_green_y,self.diff_green0_x))
		covariant_derivative_green_yy = diff_green_yy + torch.mul(torch.div(green,epsilon+self.green0),self.diff_green0_yy) + torch.mul(torch.mul(torch.div(2.,epsilon+self.green0),self.diff_green0_y),diff_green_y)
		

		covariant_derivative_blue_xx = diff_blue_xx + torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xx) + torch.mul(torch.mul(torch.div(2.,epsilon+self.blue0),self.diff_blue0_x),diff_blue_x)
		covariant_derivative_blue_xy = diff_blue_xy + torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_xy) + torch.mul(torch.div(1.,epsilon+self.blue0), torch.mul(diff_blue_x,self.diff_blue0_y)+ torch.mul(diff_blue_y,self.diff_blue0_x))
		covariant_derivative_blue_yy = diff_blue_yy + torch.mul(torch.div(blue,epsilon+self.blue0),self.diff_blue0_yy) + torch.mul(torch.mul(torch.div(2.,epsilon+self.blue0),self.diff_blue0_y),diff_blue_y)

		
		g11 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_red_x)+torch.square(covariant_derivative_green_x)+torch.square(covariant_derivative_blue_x)) #[HxW]
		g22 = torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor) + self.beta*(torch.square(covariant_derivative_red_y)+torch.square(covariant_derivative_green_y)+torch.square(covariant_derivative_blue_y)) #[HxW]
		g12 = self.beta*(torch.mul(covariant_derivative_red_x,covariant_derivative_red_y)+torch.mul(covariant_derivative_green_x,covariant_derivative_green_y)+torch.mul(covariant_derivative_blue_x,covariant_derivative_blue_y)) #[HxW]
		detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[HxW]


		invdetg = torch.div(torch.ones((self.height,self.width)).type(torch.cuda.FloatTensor),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)


		norm_regularizer1_squared = torch.mul(invg11,torch.square(covariant_derivative_red_x)+torch.square(covariant_derivative_green_x)+torch.square(covariant_derivative_blue_x)) \
									+2*torch.mul(invg12,torch.mul(covariant_derivative_red_x,covariant_derivative_red_y)+torch.mul(covariant_derivative_green_x,covariant_derivative_green_y)+torch.mul(covariant_derivative_blue_x,covariant_derivative_blue_y)) \
									+ torch.mul(invg22,torch.square(covariant_derivative_red_y)+torch.square(covariant_derivative_green_y)+torch.square(covariant_derivative_blue_y))

		norm_regularizer2_squared = torch.mul(torch.square(invg11),torch.square(covariant_derivative_red_xx)+torch.square(covariant_derivative_green_xx)+torch.square(covariant_derivative_blue_xx)) \
									+4.*torch.mul(torch.mul(invg11,invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_xy)) \
									+2.*torch.mul(torch.mul(invg11,invg22),torch.square(covariant_derivative_red_xy)+torch.square(covariant_derivative_green_xy)+torch.square(covariant_derivative_blue_xy)) \
									+4.*torch.mul(torch.mul(invg22,invg12),torch.mul(covariant_derivative_red_yy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_yy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_yy,covariant_derivative_blue_xy)) \
									+torch.mul(torch.square(invg22),torch.square(covariant_derivative_red_yy)+torch.square(covariant_derivative_green_yy)+torch.square(covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xx,covariant_derivative_red_yy)+torch.mul(covariant_derivative_green_xx,covariant_derivative_green_yy)+torch.mul(covariant_derivative_blue_xx,covariant_derivative_blue_yy)) \
									+2.*torch.mul(torch.square(invg12),torch.mul(covariant_derivative_red_xy,covariant_derivative_red_xy)+torch.mul(covariant_derivative_green_xy,covariant_derivative_green_xy)+torch.mul(covariant_derivative_blue_xy,covariant_derivative_blue_xy))
		

		norm_regularizer1 = torch.sqrt(epsilon + norm_regularizer1_squared) #[HxW]
		norm_regularizer1 = torch.unsqueeze(norm_regularizer1,dim=0) #[1xHxW]
	
		norm_regularizer2 = torch.sqrt(epsilon + norm_regularizer2_squared) #[HxW]
		norm_regularizer2 = torch.unsqueeze(norm_regularizer2,dim=0) #[1xHxW]


		return output, norm_regularizer1, norm_regularizer2'''

