import torch
import torch.nn as nn
from math import sqrt
import torch.nn.init
from .common import *
from models import *


def derivative_forward_x_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_x_forward = u[1:,:]-u[:-1,:] #[H-1xW]
	u_x_forward = torch.cat((u_x_forward,torch.zeros(1,N).type(dtype)),axis=0) #[HxW]
	return u_x_forward

def derivative_backward_x_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_x_backward = u[1:,:]-u[:-1,:] #[H-1xW]
	u_x_backward = torch.cat((torch.zeros(1,N).type(dtype),u_x_backward),axis=0) #[HxW]
	return u_x_backward

def derivative_central_x_greylevel(u,dtype):

	return 0.5*(derivative_forward_x_greylevel(u,dtype) + derivative_backward_x_greylevel(u,dtype))

def derivative_forward_y_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_y_forward = u[:,1:]-u[:,:-1] #[HxW-1]
	u_y_forward = torch.cat((u_y_forward,torch.zeros(M,1).type(dtype)),axis=1) #[HxW]
	return u_y_forward

def derivative_backward_y_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_y_backward = u[:,1:]-u[:,:-1] #[H-1xW]
	u_y_backward = torch.cat((torch.zeros(M,1).type(dtype),u_y_backward),axis=1) #[HxW]
	return u_y_backward

def derivative_central_y_greylevel(u,dtype):

	return 0.5*(derivative_forward_y_greylevel(u,dtype) + derivative_backward_y_greylevel(u,dtype))	

def derivative_xx_greylevel(u,dtype):

	return derivative_forward_x_greylevel(u,dtype) - derivative_backward_x_greylevel(u,dtype) 	

def derivative_yy_greylevel(u,dtype):

	return derivative_forward_y_greylevel(u,dtype) - derivative_backward_y_greylevel(u,dtype) 


def derivative_xy_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_plus_plus = u[1:,1:] #[H-1xW-1]
	u_plus_plus = torch.cat((u_plus_plus,torch.zeros(1,N-1).type(dtype)),axis=0) #[H-1xW]
	u_plus_plus = torch.cat((u_plus_plus,torch.zeros(M,1).type(dtype)),axis=1)  #[HxW]

	'''u_plus_plus = torch.cat((torch.zeros(1,N-1).type(dtype),u_plus_plus),axis=0) #[HxW-1]
	u_plus_plus = torch.cat((torch.zeros(M,1).type(dtype),u_plus_plus),axis=1)'''  #[HxW]

	u_minus_minus = u[:-1,:-1] #[H-1xW-1]
	u_minus_minus = torch.cat((torch.zeros(1,N-1).type(dtype),u_minus_minus),axis=0) #[H-1xW]
	u_minus_minus = torch.cat((torch.zeros(M,1).type(dtype),u_minus_minus),axis=1)  #[HxW]

	'''u_minus_minus = torch.cat((u_minus_minus,torch.zeros(1,N-1).type(dtype)),axis=0) #[HxW-1]
	u_minus_minus = torch.cat((u_minus_minus,torch.zeros(M,1).type(dtype)),axis=1)'''  #[HxW]

	u_plus_minus = u[1:,:-1] #[H-1xW-1]	
	u_plus_minus = torch.cat((u_plus_minus,torch.zeros(1,N-1).type(dtype)),axis=0) #[H-1xW]
	u_plus_minus = torch.cat((torch.zeros(M,1).type(dtype),u_plus_minus),axis=1) #[HxW]

	'''u_plus_minus = torch.cat((torch.zeros(1,N-1).type(dtype),u_plus_minus),axis=0) #[HxW-1]
	u_plus_minus = torch.cat((u_plus_minus, torch.zeros(M,1).type(dtype)),axis=1)''' #[HxW]

	u_minus_plus = u[:-1,1:] #[H-1xW-1]
	u_minus_plus = torch.cat((torch.zeros(1,N-1).type(dtype),u_minus_plus),axis=0) #[H-1xW]
	u_minus_plus = torch.cat((u_minus_plus,torch.zeros(M,1).type(dtype)),axis=1) #[HxW]
	'''u_minus_plus = torch.cat((u_minus_plus,torch.zeros(1,N-1).type(dtype),),axis=0) #[HxW-1]
	u_minus_plus = torch.cat((torch.zeros(M,1).type(dtype),u_minus_plus),axis=1) #[HxW]'''

														
	u_xy = (u_plus_plus+u_minus_minus-u_plus_minus-u_minus_plus)/4.	
	return u_xy

def derivative_xxx_greylevel(u,dtype):

	return derivative_central_x_greylevel(derivative_xx_greylevel(u,dtype),dtype)

def derivative_yyy_greylevel(u,dtype):

	return derivative_central_y_greylevel(derivative_yy_greylevel(u,dtype),dtype)	


def derivative_xxy_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_plus2_plus1 = u[2:,1:] #[H-2xW-1]
	u_plus2_plus1 = torch.cat((u_plus2_plus1,torch.zeros(2,N-1).type(dtype)),axis=0) #[HxW-1]
	u_plus2_plus1 = torch.cat((u_plus2_plus1,torch.zeros(M,1).type(dtype)),axis=1)   #[HxW]
	
	u_0_plus1 = u[:,1:] #[HxW-1]
	u_0_plus1 = torch.cat((u_0_plus1,torch.zeros(M,1).type(dtype)),axis=1) #[HxW]

	u_minus2_plus1 = u[:-2,1:] #[H-2xW-1]
	u_minus2_plus1 = torch.cat((torch.zeros(2,N-1).type(dtype),u_minus2_plus1),axis=0) #[HxW-1]
	u_minus2_plus1 = torch.cat((u_minus2_plus1,torch.zeros(M,1).type(dtype)),axis=1) #[HxW]


	u_plus2_minus1 = u[2:,:-1] #[H-2xW-1]
	u_plus2_minus1 = torch.cat((u_plus2_minus1,torch.zeros(2,N-1).type(dtype)),axis=0) #[HxW-1]
	u_plus2_minus1 = torch.cat((torch.zeros(M,1).type(dtype),u_plus2_minus1),axis=1) #[HxW]


	u_0_minus1 = u[:,:-1] #[HxW-1]
	u_0_minus1 = torch.cat((torch.zeros(M,1).type(dtype),u_0_minus1),axis=1) #[HxW]


	u_minus2_minus1 = u[:-2,:-1] #[H-2xW-1]
	u_minus2_minus1 = torch.cat((torch.zeros(2,N-1).type(dtype),u_minus2_minus1),axis=0) #[HxW-1]
	u_minus2_minus1 = torch.cat((torch.zeros(M,1).type(dtype),u_minus2_minus1),axis=1) #[HxW]

	u_xxy = (u_plus2_plus1-2.*u_0_plus1+u_minus2_plus1-u_plus2_minus1+2.*u_0_minus1-u_minus2_minus1)/8.

	return u_xxy


def derivative_xyy_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_plus1_plus2 = u[1:,2:] #[H-1xW-2]
	u_plus1_plus2 = torch.cat((u_plus1_plus2,torch.zeros(M-1,2).type(dtype)),axis=1) #[H-1xW]
	u_plus1_plus2 = torch.cat((u_plus1_plus2,torch.zeros(1,N).type(dtype)),axis=0)   #[HxW]

	u_plus1_0 = u[1:,:] #[H-1xW]
	u_plus1_0 = torch.cat((u_plus1_0,torch.zeros(1,N).type(dtype)),axis=0) #[HxW]


	u_plus1_minus2 = u[1:,:-2] #[H-1xW-2]
	u_plus1_minus2 = torch.cat((torch.zeros(M-1,2).type(dtype),u_plus1_minus2),axis=1) #[H-1xW]
	u_plus1_minus2 = torch.cat((u_plus1_minus2,torch.zeros(1,N).type(dtype)),axis=0)   #[HxW]

	u_minus1_plus2 = u[:-1,2:] #[H-1xW-2]
	u_minus1_plus2 = torch.cat((u_minus1_plus2,torch.zeros(M-1,2).type(dtype)),axis=1) #[H-1xW]
	u_minus1_plus2 = torch.cat((torch.zeros(1,N).type(dtype),u_minus1_plus2),axis=0)   #[HxW]

	u_minus1_0 = u[:-1,:] #[H-1xW]
	u_minus1_0 = torch.cat((torch.zeros(1,N).type(dtype),u_minus1_0),axis=0) #[HxW]
	
	u_minus1_minus2 = u[:-1,:-2] #[H-1xW-2]
	u_minus1_minus2 = torch.cat((torch.zeros(M-1,2).type(dtype),u_minus1_minus2),axis=1) #[H-1xW]
	u_minus1_minus2 = torch.cat((torch.zeros(1,N).type(dtype),u_minus1_minus2),axis=0) #[HxW]



	u_xyy = (u_plus1_plus2-2.*u_plus1_0+u_plus1_minus2-u_minus1_plus2+2.*u_minus1_0-u_minus1_minus2)/8.

	return u_xyy	




def riemannianmetric_greylevel(u,beta,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_x = derivative_central_x_greylevel(u,dtype)
	u_y = derivative_central_y_greylevel(u,dtype)

	g11 = torch.ones((M,N)).type(dtype) + beta*torch.square(u_x)
	g12 = beta*torch.mul(u_x,u_y)
	g22 = torch.ones((M,N)).type(dtype) + beta*torch.square(u_y)

	return g11,g12,g22


def riemannianmetric_color(u,beta,dtype):

	M = u.size(dim=1)
	N = u.size(dim=2)

	u0_x = derivative_central_x_greylevel(u[0,:,:],dtype)
	u0_y = derivative_central_y_greylevel(u[0,:,:],dtype)

	u1_x = derivative_central_x_greylevel(u[1,:,:],dtype)
	u1_y = derivative_central_y_greylevel(u[1,:,:],dtype)

	u2_x = derivative_central_x_greylevel(u[2,:,:],dtype)
	u2_y = derivative_central_y_greylevel(u[2,:,:],dtype)



	g11 = torch.ones((M,N)).type(dtype) + beta*(torch.square(u0_x)+torch.square(u1_x)+torch.square(u2_x))
	g12 = beta*(torch.mul(u0_x,u0_y)+torch.mul(u1_x,u1_y)+torch.mul(u2_x,u2_y))
	g22 = torch.ones((M,N)).type(dtype) + beta*(torch.square(u0_y)+torch.square(u1_y)+torch.square(u2_y))

	return g11,g12,g22


def riemanniangradient_greylevel(u,g11,g12,g22,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	detg = torch.mul(g11,g22) - torch.square(g12)

	invdetg = torch.div(torch.ones((M,N)).type(dtype),detg)
	invg11 = torch.mul(invdetg,g22)
	invg12 = - torch.mul(invdetg,g12)
	invg22 = torch.mul(invdetg,g11)

	u_x_forward = u[1:,:]-u[:-1,:] #[H-1xW]
	u_x_forward = torch.cat((u_x_forward,torch.zeros(1,N).type(dtype)),axis=0) #[HxW]

	u_y_forward = u[:,1:]-u[:,:-1] #[HxW-1]
	u_y_forward = torch.cat((u_y_forward,torch.zeros(M,1).type(dtype)),axis=1) #[HxW]

	return torch.mul(invg11,u_x_forward) + torch.mul(invg12,u_y_forward), torch.mul(invg12,u_x_forward) + torch.mul(invg22,u_y_forward)



def riemanniandivergence_greylevel(U1,U2,g11,g12,g22,dtype):

	M = U1.size(dim=0)
	N = U2.size(dim=1)

	sqrt_abs_detg = torch.sqrt(torch.abs(torch.mul(g11,g22) - torch.square(g12)))
	component1 = torch.mul(sqrt_abs_detg,U1)
	component2 = torch.mul(sqrt_abs_detg,U2)

	component1_x_backward = component1[1:,:]-component1[:-1,:]
	component1_x_backward = torch.cat((torch.zeros(1,N).type(dtype),component1_x_backward),axis=0)

	component2_y_backward = component2[:,1:]-component2[:,:-1]
	component2_y_backward = torch.cat((torch.zeros(M,1).type(dtype),component2_y_backward),axis=1)

	return torch.mul(torch.div(torch.ones(M,N).type(dtype),sqrt_abs_detg),component1_x_backward+component2_y_backward)


def nonlineardiracoperator(u1,u2,dtype):


	M = u1.size(dim=0)
	N = u1.size(dim=1)

	u1_x = derivative_central_x_greylevel(u1,dtype)
	u1_y = derivative_central_y_greylevel(u1,dtype)

	u2_x = derivative_central_x_greylevel(u2,dtype)
	u2_y = derivative_central_y_greylevel(u2,dtype)

	cross_product_12 = torch.mul(u1_x,u2_y)-torch.mul(u2_x,u1_y)

	return u1_x+torch.mul(torch.sign(cross_product_12),u2_y),torch.mul(torch.sign(cross_product_12),u2_x)-u1_y



def orthogonal_moving_frame(g11,g12,g22,dtype):

	M = g11.size(dim=0)
	N = g11.size(dim=1)

	detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[1x1xHxW]

	lambda_plus = (1./2.)*(g11+g22+torch.sqrt(0.00001*torch.ones((M,N)).type(dtype)+torch.square(g11+g22)-4.*detg))
	lambda_moins = (1./2.)*(g11+g22-torch.sqrt(0.00001*torch.ones((M,N)).type(dtype)+torch.square(g11+g22)-4.*detg))

	a = torch.div(lambda_plus-g22,0.00001*torch.ones((M,N)).type(dtype)+torch.mul(torch.sqrt(lambda_plus),torch.sqrt(torch.square(g12)+torch.square(lambda_plus-g22))))
	b = torch.div(g12,0.00001*torch.ones((M,N)).type(dtype)+torch.mul(torch.sqrt(lambda_plus),torch.sqrt(torch.square(g12)+torch.square(lambda_plus-g22))))
	c = torch.mul(torch.sign(g12), 0.00001*torch.ones((M,N)).type(dtype)+torch.div(lambda_moins-g22,torch.mul(torch.sqrt(lambda_moins),torch.sqrt(torch.square(g12)+torch.square(lambda_moins-g22)))))
	d = torch.mul(torch.sign(g12), 0.00001*torch.ones((M,N)).type(dtype)+torch.div(g12,torch.mul(torch.sqrt(lambda_moins),torch.sqrt(torch.square(g12)+torch.square(lambda_moins-g22)))))

 
	'''print(torch.max(lambda_plus.detach()))
	print(torch.max(lambda_moins.detach()))
	print(torch.min(lambda_plus.detach()))
	print(torch.min(lambda_moins.detach()))

	print(torch.max( (torch.square(a)+torch.square(b)).detach()))
	print(torch.min( (torch.square(a)+torch.square(b)).detach()))
	print(torch.max( (torch.square(c)+torch.square(d)).detach()))
	print(torch.min( (torch.square(c)+torch.square(d)).detach() ))
	print(torch.max( (torch.mul(a,c)+torch.mul(b,d)).detach() ))
	print(torch.min( (torch.mul(a,c)+torch.mul(b,d)).detach() ))'''


	return a,b,c,d



def laplacebeltrami_greylevel(u,g11,g12,g22,dtype):

	U1,U2 = riemanniangradient_greylevel(u,g11,g12,g22,dtype)

	return riemanniandivergence_greylevel(U1,U2,g11,g12,g22,dtype)


def gaussiancurvature_greylevel(u,beta,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)
	
	u_x = derivative_central_x_greylevel(u)
	u_y = derivative_central_y_greylevel(u)

	u_xx = derivative_xx_greylevel(u)
	u_xy = derivative_xy_greylevel(u)
	u_yy = derivative_yy_greylevel(u)

	num = torch.mul(u_xx,u_yy)-torch.square(u_xy)
	den = (beta**(3/2))*torch.square((1./beta)*torch.ones(M,N).type(dtype) + torch.square(u_x) + torch.square(u_y))
																		 
	return torch.div(num,den)


def meancurvature_greylevel(u,beta,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_x = derivative_central_x_greylevel(u)
	u_y = derivative_central_y_greylevel(u)

	u_xx = derivative_xx_greylevel(u)
	u_xy = derivative_xy_greylevel(u)
	u_yy = derivative_yy_greylevel(u)

	num = torch.mul(u_xx,torch.ones(M,N).type(dtype)+beta*(torch.square(u_y))) \
		-2.*(beta)*torch.mul(torch.mul(u_xy,u_x),u_y) \
		+ torch.mul(u_yy,torch.ones(M,N).type(dtype)+beta*torch.square(u_x))

	den = 2.*(beta**(2/3))*torch.pow((1./(beta))*torch.ones(M,N).type(dtype)+torch.square(u_x)+torch.square(u_y), 3.2)


	return torch.div(num,den)



def Levi_Cevita_connection_coefficients_greylevel_standard_frame(u,beta,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	
	g11,g12,g22 = riemannianmetric_greylevel(u,beta,dtype)

	detg = torch.mul(g11,g22) - torch.mul(g12,g12) 
	invdetg = torch.div(torch.ones(M,N).type(dtype),detg)


	u_x = derivative_central_x_greylevel(u,dtype)
	u_y = derivative_central_y_greylevel(u,dtype)
	u_xx = derivative_xx_greylevel(u,dtype)
	u_xy = derivative_xy_greylevel(u,dtype)
	u_yy = derivative_yy_greylevel(u,dtype)


	Gamma1_11 = beta*torch.mul(invdetg,torch.mul(u_x,u_xx)) # [HxW]
	Gamma2_22 = beta*torch.mul(invdetg,torch.mul(u_y,u_yy)) # [HxW]
	Gamma1_12 = beta*torch.mul(invdetg,torch.mul(u_x,u_xy)) # [HxW]
	Gamma1_21 = Gamma1_12 # [1x1xHxW]
	Gamma2_12 = beta*torch.mul(invdetg,torch.mul(u_y,u_xy)) # [HxW]
	Gamma2_21 = Gamma2_12 # [1x1xHxW]
	Gamma1_22 = beta*torch.mul(invdetg,torch.mul(u_x,u_yy)) # [HxW]
	Gamma2_11 = beta*torch.mul(invdetg,torch.mul(u_y,u_xx)) # [HxW]

	return Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11

def Levi_Cevita_connection_coefficients_color_standard_frame(u,beta,dtype):

	M = u.size(dim=1)
	N = u.size(dim=2)

	
	g11,g12,g22 = riemannianmetric_color(u,beta,dtype)

	detg = torch.mul(g11,g22) - torch.mul(g12,g12) 
	invdetg = torch.div(torch.ones(M,N).type(dtype),detg)
	invg11 = torch.mul(invdetg,g22)
	invg12 = - torch.mul(invdetg,g12)
	invg22 = torch.mul(invdetg,g11)

	g11_x = derivative_central_x_greylevel(g11,dtype)
	g11_y = derivative_central_y_greylevel(g11,dtype)

	g12_x = derivative_central_x_greylevel(g12,dtype)
	g12_y = derivative_central_y_greylevel(g12,dtype)

	g22_x = derivative_central_x_greylevel(g22,dtype)
	g22_y = derivative_central_y_greylevel(g22,dtype)
	



	Gamma1_11 =  0.5*(torch.mul(invg11,g11_x) + torch.mul(invg12,2.*g12_x-g11_y)) # [HxW]
	Gamma2_22 =  0.5*(torch.mul(invg22,g22_y) + torch.mul(invg12,2.*g12_y-g22_x)) # [HxW]
	Gamma1_12 =  0.5*(torch.mul(invg11,g11_y)+torch.mul(invg12,g22_x)) # [HxW]
	Gamma1_21 = Gamma1_12 # [HxW]
	Gamma2_12 =  0.5*(torch.mul(invg22,g22_x)+torch.mul(invg12,g11_y)) # [HxW]
	Gamma2_21 = Gamma2_12 # [HxW]
	Gamma1_22 =  0.5*(torch.mul(invg12,g22_y) + torch.mul(invg11,2.*g12_y-g22_x)) # [HxW]
	Gamma2_11 =  0.5*(torch.mul(invg12,g11_x) + torch.mul(invg22,2.*g12_x-g11_y))# [HxW]

	return Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11


def norm_first_order_covariant_derivative_greylevel(u,epsilon,beta,dtype,invg11,invg12,invg22):

	M = u.size(dim=0)
	N = u.size(dim=1)

	
	diff_img_x = derivative_central_x_greylevel(u,dtype)
	diff_img_y = derivative_central_y_greylevel(u,dtype)

	norm_first_order_covariant_derivative_squared = torch.mul(invg11,torch.square(diff_img_x)) \
									+2*torch.mul(invg12,torch.mul(diff_img_x,diff_img_y)) \
									+ torch.mul(invg22,torch.square(diff_img_y))

	norm_first_order_covariant_derivative = torch.sqrt(epsilon*torch.ones(M,N).type(dtype) + norm_first_order_covariant_derivative_squared)

	return norm_first_order_covariant_derivative								

def norm_first_order_covariant_derivative_color(u,epsilon,beta,dtype,invg11,invg12,invg22):

	M = u.size(dim=1)
	N = u.size(dim=2)

	
	diff_img0_x = derivative_central_x_greylevel(u[0,:,:],dtype)
	diff_img0_y = derivative_central_y_greylevel(u[0,:,:],dtype)

	diff_img1_x = derivative_central_x_greylevel(u[1,:,:],dtype)
	diff_img1_y = derivative_central_y_greylevel(u[1,:,:],dtype)

	diff_img2_x = derivative_central_x_greylevel(u[2,:,:],dtype)
	diff_img2_y = derivative_central_y_greylevel(u[2,:,:],dtype)

	norm_first_order_covariant_derivative_squared = torch.mul(invg11,torch.square(diff_img0_x)+torch.square(diff_img1_x)+torch.square(diff_img2_x)) \
									+2*torch.mul(invg12,torch.mul(diff_img0_x,diff_img0_y)+torch.mul(diff_img1_x,diff_img1_y)+torch.mul(diff_img2_x,diff_img2_y)) \
									+ torch.mul(invg22,torch.square(diff_img0_y)+torch.square(diff_img1_y)+torch.square(diff_img2_y))

	norm_first_order_covariant_derivative = torch.sqrt(epsilon*torch.ones(M,N).type(dtype) + norm_first_order_covariant_derivative_squared)

	return norm_first_order_covariant_derivative	


def norm_second_order_covariant_derivative_greylevel(u,epsilon,beta,dtype,invg11,invg12,invg22,Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11):


	M = u.size(dim=0)
	N = u.size(dim=1)

	
	diff_img_x = derivative_central_x_greylevel(u,dtype)
	diff_img_y = derivative_central_y_greylevel(u,dtype)

	diff_img_xx = derivative_xx_greylevel(u,dtype)
	diff_img_xy = derivative_xy_greylevel(u,dtype)
	diff_img_yy = derivative_yy_greylevel(u,dtype)

	terms_xx = diff_img_xx - torch.mul(Gamma1_11,diff_img_x)-torch.mul(Gamma2_11,diff_img_y)
	terms_xy = diff_img_xy - torch.mul(Gamma1_12,diff_img_x)-torch.mul(Gamma2_12,diff_img_y)
	terms_yy = diff_img_yy - torch.mul(Gamma1_22,diff_img_x)-torch.mul(Gamma2_22,diff_img_y)

	norm_second_order_covariant_derivative_squared = torch.mul(torch.square(invg11),torch.square(terms_xx)) \
						+ 4.*torch.mul(invg11,torch.mul(invg12,torch.mul(terms_xx,terms_xy))) \
						+ 2.*torch.mul(invg11,torch.mul(invg22,torch.mul(terms_xy,terms_xy))) \
						+ 4.*torch.mul(invg12,torch.mul(invg22,torch.mul(terms_xy,terms_yy))) \
						+ torch.mul(torch.square(invg22),torch.square(terms_yy)) \
						+ 2.*torch.mul(torch.square(invg12),torch.mul(terms_xy,terms_xy)) \
						+ 2.*torch.mul(torch.square(invg12),torch.mul(terms_xx,terms_yy))
												

	norm_second_order_covariant_derivative = torch.sqrt(epsilon*torch.ones(M,N).type(dtype) + norm_second_order_covariant_derivative_squared)

	return norm_second_order_covariant_derivative

def norm_second_order_covariant_derivative_color(u,epsilon,beta,dtype,invg11,invg12,invg22,Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11):


	M = u.size(dim=1)
	N = u.size(dim=2)

	
	diff_img0_x = derivative_central_x_greylevel(u[0,:,:],dtype)
	diff_img0_y = derivative_central_y_greylevel(u[0,:,:],dtype)

	diff_img1_x = derivative_central_x_greylevel(u[1,:,:],dtype)
	diff_img1_y = derivative_central_y_greylevel(u[1,:,:],dtype)

	diff_img2_x = derivative_central_x_greylevel(u[2,:,:],dtype)
	diff_img2_y = derivative_central_y_greylevel(u[2,:,:],dtype)

	diff_img0_xx = derivative_xx_greylevel(u[0,:,:],dtype)
	diff_img0_xy = derivative_xy_greylevel(u[0,:,:],dtype)
	diff_img0_yy = derivative_yy_greylevel(u[0,:,:],dtype)

	diff_img1_xx = derivative_xx_greylevel(u[1,:,:],dtype)
	diff_img1_xy = derivative_xy_greylevel(u[1,:,:],dtype)
	diff_img1_yy = derivative_yy_greylevel(u[1,:,:],dtype)

	diff_img2_xx = derivative_xx_greylevel(u[2,:,:],dtype)
	diff_img2_xy = derivative_xy_greylevel(u[2,:,:],dtype)
	diff_img2_yy = derivative_yy_greylevel(u[2,:,:],dtype)

	terms0_xx = diff_img0_xx - torch.mul(Gamma1_11,diff_img0_x)-torch.mul(Gamma2_11,diff_img0_y)
	terms0_xy = diff_img0_xy - torch.mul(Gamma1_12,diff_img0_x)-torch.mul(Gamma2_12,diff_img0_y)
	terms0_yy = diff_img0_yy - torch.mul(Gamma1_22,diff_img0_x)-torch.mul(Gamma2_22,diff_img0_y)

	terms1_xx = diff_img1_xx - torch.mul(Gamma1_11,diff_img1_x)-torch.mul(Gamma2_11,diff_img1_y)
	terms1_xy = diff_img1_xy - torch.mul(Gamma1_12,diff_img1_x)-torch.mul(Gamma2_12,diff_img1_y)
	terms1_yy = diff_img1_yy - torch.mul(Gamma1_22,diff_img1_x)-torch.mul(Gamma2_22,diff_img1_y)

	terms2_xx = diff_img2_xx - torch.mul(Gamma1_11,diff_img2_x)-torch.mul(Gamma2_11,diff_img2_y)
	terms2_xy = diff_img2_xy - torch.mul(Gamma1_12,diff_img2_x)-torch.mul(Gamma2_12,diff_img2_y)
	terms2_yy = diff_img2_yy - torch.mul(Gamma1_22,diff_img2_x)-torch.mul(Gamma2_22,diff_img2_y)



	norm_second_order_covariant_derivative_squared = torch.mul(torch.square(invg11),torch.square(terms0_xx)+torch.square(terms1_xx)+torch.square(terms2_xx)) \
						+ 4.*torch.mul(invg11,torch.mul(invg12,torch.mul(terms0_xx,terms0_xy)+torch.mul(terms1_xx,terms1_xy)+torch.mul(terms2_xx,terms2_xy))) \
						+ 2.*torch.mul(invg11,torch.mul(invg22,torch.mul(terms0_xy,terms0_xy)+torch.mul(terms1_xy,terms1_xy)+torch.mul(terms2_xy,terms2_xy))) \
						+ 4.*torch.mul(invg12,torch.mul(invg22,torch.mul(terms0_xy,terms0_yy)+torch.mul(terms1_xy,terms1_yy)+torch.mul(terms2_xy,terms2_yy))) \
						+ torch.mul(torch.square(invg22),torch.square(terms0_yy)+torch.square(terms1_yy)+torch.square(terms2_yy)) \
						+ 2.*torch.mul(torch.square(invg12),torch.mul(terms0_xy,terms0_xy)+torch.mul(terms1_xy,terms1_xy)+torch.mul(terms2_xy,terms2_xy)) \
						+ 2.*torch.mul(torch.square(invg12),torch.mul(terms0_xx,terms0_yy)+torch.mul(terms1_xx,terms1_yy)+torch.mul(terms2_xx,terms2_yy))
												

	norm_second_order_covariant_derivative = torch.sqrt(epsilon*torch.ones(M,N).type(dtype) + norm_second_order_covariant_derivative_squared)

	return norm_second_order_covariant_derivative


def norm_third_order_covariant_derivative_greylevel(u,epsilon,beta,dtype,invg11,invg12,invg22,Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11):	

	
	M = u.size(dim=0)
	N = u.size(dim=1)

	diff_img_x = derivative_central_x_greylevel(u,dtype)     #[HxW]
	diff_img_y = derivative_central_y_greylevel(u,dtype)     #[HxW]

	diff_img_xx = derivative_xx_greylevel(u,dtype) #[HxW]
	diff_img_xy = derivative_xy_greylevel(u,dtype) #[HxW]
	diff_img_yy = derivative_yy_greylevel(u,dtype) #[HxW]

	diff_img_xxx = derivative_xxx_greylevel(u,dtype) #[HxW]
	diff_img_xxy = derivative_xxy_greylevel(u,dtype) #[HxW]
	diff_img_xyy = derivative_xyy_greylevel(u,dtype) #[HxW]
	diff_img_yyy = derivative_yyy_greylevel(u,dtype) #[HxW]


	terms_xxx = diff_img_xxx \
				- 3.*(torch.mul(Gamma1_11,diff_img_xx)+torch.mul(Gamma2_11,diff_img_xy)) \
				+ 2.*(torch.mul(torch.square(Gamma1_11),diff_img_x) + torch.mul(Gamma2_11,torch.mul(Gamma1_12,diff_img_x)) + torch.mul(Gamma1_11,torch.mul(Gamma2_11,diff_img_y)) + torch.mul(Gamma2_11,torch.mul(Gamma2_12,diff_img_y))) \
				- torch.mul(derivative_central_x_greylevel(Gamma1_11,dtype),diff_img_x) - torch.mul(derivative_central_x_greylevel(Gamma2_11,dtype),diff_img_y)


	terms_yyy = diff_img_yyy  \
				- 3.*(torch.mul(Gamma1_22,diff_img_xy)+torch.mul(Gamma2_22,diff_img_yy)) \
				+ 2.*(torch.mul(torch.square(Gamma2_22),diff_img_y) + torch.mul(Gamma1_22,torch.mul(Gamma1_12,diff_img_x)) + torch.mul(Gamma2_22,torch.mul(Gamma1_22,diff_img_x)) + torch.mul(Gamma1_22,torch.mul(Gamma2_12,diff_img_y))) \
				- torch.mul(derivative_central_y_greylevel(Gamma1_22,dtype),diff_img_x) - torch.mul(derivative_central_y_greylevel(Gamma2_22,dtype),diff_img_y) 
	

	terms_xxy = diff_img_xxy \
				- 2.*(torch.mul(Gamma1_12,diff_img_xx) + torch.mul(Gamma2_12,diff_img_xy)) - (torch.mul(Gamma1_11,diff_img_xy) + torch.mul(Gamma2_11,diff_img_yy)) \
				+ 2.*(torch.mul(Gamma1_11,torch.mul(Gamma1_12,diff_img_x))) + torch.mul(Gamma1_11,torch.mul(Gamma2_12,diff_img_y)) + torch.mul(Gamma2_11,torch.mul(Gamma1_22,diff_img_x))+torch.mul(Gamma2_12,torch.mul(Gamma1_12,diff_img_x)) \
				+ torch.mul(Gamma2_11,torch.mul(Gamma2_22,diff_img_y)) + torch.mul(torch.square(Gamma2_12),diff_img_y) + torch.mul(Gamma1_12,torch.mul(Gamma2_11,diff_img_y)) \
				- torch.mul(derivative_central_x_greylevel(Gamma1_12,dtype),diff_img_x) - torch.mul(derivative_central_x_greylevel(Gamma2_12,dtype),diff_img_y)  	
				

	terms_yxx = diff_img_xxy \
				-2.*( torch.mul(Gamma1_12,diff_img_xx) + torch.mul(Gamma2_12,diff_img_xy)) - (torch.mul(Gamma1_11,diff_img_xy) + torch.mul(Gamma2_11,diff_img_yy)) \
				+ 2.*torch.mul(Gamma1_11,torch.mul(Gamma1_12,diff_img_x) + torch.mul(Gamma2_11,torch.mul(Gamma1_12,diff_img_y)) + torch.mul(Gamma2_21,torch.mul(Gamma1_12,diff_img_x)) + torch.mul(torch.square(Gamma2_12),diff_img_y)) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_11,dtype),diff_img_x) - torch.mul(derivative_central_y_greylevel(Gamma2_11,dtype),diff_img_y)	

	terms_xyx =	diff_img_xxy \
				-2.*( torch.mul(Gamma2_12,diff_img_xy) + torch.mul(Gamma1_12,diff_img_xx)) - (torch.mul(Gamma1_11,diff_img_xy)+torch.mul(Gamma2_11,diff_img_yy)) \
				+ 2.*torch.mul(Gamma1_11,torch.mul(Gamma1_12,diff_img_x)) + torch.mul(Gamma2_12,torch.mul(Gamma1_12,diff_img_x)) + torch.mul(Gamma2_11,torch.mul(Gamma1_22,diff_img_x))  \
				+ torch.mul(Gamma1_12,torch.mul(Gamma2_11,diff_img_y)) + torch.mul(Gamma1_11,torch.mul(Gamma2_21,diff_img_y)) + torch.mul(Gamma2_11,torch.mul(Gamma2_22,diff_img_y)) + torch.mul(torch.square(Gamma2_12),diff_img_y) \
				-  torch.mul(derivative_central_x_greylevel(Gamma1_21,dtype),diff_img_x) - torch.mul(derivative_central_x_greylevel(Gamma2_21,dtype),diff_img_y)

	terms_yyx = diff_img_xyy \
				- 2.*(torch.mul(Gamma1_21,diff_img_xy) + torch.mul(Gamma2_21,diff_img_yy)) - (torch.mul(Gamma1_22,diff_img_xx) + torch.mul(Gamma2_22,diff_img_xy)) \
				+ 2.*(torch.mul(Gamma2_22,torch.mul(Gamma2_21,diff_img_y))) + torch.mul(Gamma1_22,torch.mul(Gamma1_11,diff_img_x)) + torch.mul(Gamma1_22,torch.mul(Gamma2_11,diff_img_y)) + torch.mul(torch.square(Gamma1_21),diff_img_x) \
				+ torch.mul(Gamma2_22,torch.mul(Gamma1_21,diff_img_x)) + torch.mul(Gamma1_21,torch.mul(Gamma2_21,diff_img_y)) + torch.mul(Gamma2_21,torch.mul(Gamma1_22,diff_img_x)) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_21,dtype),diff_img_x) - torch.mul(derivative_central_y_greylevel(Gamma2_21,dtype),diff_img_y)

	
	terms_xyy = diff_img_xyy \
				- 2.*(torch.mul(Gamma1_12,diff_img_xy) + torch.mul(Gamma2_12,diff_img_yy)) - (torch.mul(Gamma2_22,diff_img_xy) + torch.mul(Gamma1_22,diff_img_xx)) \
				+ 2.*torch.mul(Gamma1_12,torch.mul(Gamma2_12,diff_img_y) + torch.mul(Gamma2_12,torch.mul(Gamma1_22,diff_img_x)) + torch.mul(Gamma2_22,torch.mul(Gamma2_12,diff_img_y)) + torch.mul(torch.square(Gamma1_12),diff_img_x)) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_22,dtype),diff_img_x) - torch.mul(derivative_central_y_greylevel(Gamma2_22,dtype),diff_img_y)



	terms_yxy =	diff_img_xyy \
				-2.*( torch.mul(Gamma1_12,diff_img_xy) + torch.mul(Gamma2_12,diff_img_yy)) - (torch.mul(Gamma2_22,diff_img_xy)+torch.mul(Gamma1_22,diff_img_xx)) \
				+ 2.*torch.mul(Gamma2_22,torch.mul(Gamma2_12,diff_img_y)) + torch.mul(Gamma1_21,torch.mul(Gamma2_21,diff_img_y)) + torch.mul(Gamma2_21,torch.mul(Gamma1_22,diff_img_x))  \
				+ torch.mul(Gamma1_22,torch.mul(Gamma1_11,diff_img_x)) + torch.mul(Gamma1_22,torch.mul(Gamma2_11,diff_img_y)) + torch.mul(Gamma2_22,torch.mul(Gamma1_12,diff_img_x)) + torch.mul(torch.square(Gamma1_21),diff_img_x) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_12,dtype),diff_img_x) - torch.mul(derivative_central_y_greylevel(Gamma2_12,dtype),diff_img_y)


	norm_third_order_covariant_derivative_squared = torch.mul(torch.pow(invg11,3),torch.square(terms_xxx))+torch.mul(torch.pow(invg22,3),torch.square(terms_yyy)) +  2.*(torch.mul(torch.pow(invg12,3),torch.mul(terms_xxx,terms_yyy) + torch.mul(terms_xxy,terms_yyx)+torch.mul(terms_xyx,terms_yxy)+torch.mul(terms_xyy,terms_yxx))) + 2.*(torch.mul(torch.mul(invg11,torch.mul(invg12,invg22)),torch.mul(terms_xyx,terms_xyy)+torch.mul(terms_yxx,terms_yxy)+torch.mul(terms_xxy,terms_xyy)+torch.mul(terms_yxx,terms_yyx)+torch.mul(terms_xxy,terms_yxy)+torch.mul(terms_xyx,terms_yyx))) + 2.*(torch.mul(torch.mul(torch.square(invg11),invg12),torch.mul(terms_xxx,terms_xxy)+torch.mul(terms_xxx,terms_xyx)+torch.mul(terms_xxx,terms_yxx))) + 2.*(torch.mul(torch.mul(torch.square(invg22),invg12),torch.mul(terms_yyx,terms_yyy)+torch.mul(terms_yxy,terms_yyy)+torch.mul(terms_xyy,terms_yyy))) + 2.*(torch.mul(torch.mul(torch.square(invg12),invg11),torch.mul(terms_xxx,terms_xyy)+torch.mul(terms_xxy,terms_xyx)+torch.mul(terms_xxx,terms_yxy)+torch.mul(terms_xxy,terms_yxx)+torch.mul(terms_xxx,terms_yyx)+torch.mul(terms_xyx,terms_yxx))) + 2.*(torch.mul(torch.mul(torch.square(invg12),invg22),torch.mul(terms_yxx,terms_yyy)+torch.mul(terms_yxy,terms_yyx)+torch.mul(terms_xyx,terms_yyy)+torch.mul(terms_xyy,terms_yyx)+torch.mul(terms_xxy,terms_yyy)+torch.mul(terms_xyy,terms_yxy))) + torch.mul(torch.mul(torch.square(invg11),invg22),torch.square(terms_xxy)+torch.square(terms_xyx)+torch.square(terms_yxx)) + torch.mul(torch.mul(torch.square(invg22),invg11),torch.square(terms_xyy)+torch.square(terms_yxy)+torch.square(terms_yyx))

		 		
	norm_third_order_covariant_derivative = torch.sqrt(epsilon*torch.ones(M,N).type(dtype)+norm_third_order_covariant_derivative_squared)

	return norm_third_order_covariant_derivative	


def norm_third_order_covariant_derivative_color(u,epsilon,beta,dtype,invg11,invg12,invg22,Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11):	

	
	M = u.size(dim=1)
	N = u.size(dim=2)

	diff_img0_x = derivative_central_x_greylevel(u[0,:,:],dtype)
	diff_img0_y = derivative_central_y_greylevel(u[0,:,:],dtype)

	diff_img1_x = derivative_central_x_greylevel(u[1,:,:],dtype)
	diff_img1_y = derivative_central_y_greylevel(u[1,:,:],dtype)

	diff_img2_x = derivative_central_x_greylevel(u[2,:,:],dtype)
	diff_img2_y = derivative_central_y_greylevel(u[2,:,:],dtype)

	diff_img0_xx = derivative_xx_greylevel(u[0,:,:],dtype)
	diff_img0_xy = derivative_xy_greylevel(u[0,:,:],dtype)
	diff_img0_yy = derivative_yy_greylevel(u[0,:,:],dtype)

	diff_img1_xx = derivative_xx_greylevel(u[1,:,:],dtype)
	diff_img1_xy = derivative_xy_greylevel(u[1,:,:],dtype)
	diff_img1_yy = derivative_yy_greylevel(u[1,:,:],dtype)

	diff_img2_xx = derivative_xx_greylevel(u[2,:,:],dtype)
	diff_img2_xy = derivative_xy_greylevel(u[2,:,:],dtype)
	diff_img2_yy = derivative_yy_greylevel(u[2,:,:],dtype)

	diff_img0_xxx = derivative_xxx_greylevel(u[0,:,:],dtype) #[HxW]
	diff_img0_xxy = derivative_xxy_greylevel(u[0,:,:],dtype) #[HxW]
	diff_img0_xyy = derivative_xyy_greylevel(u[0,:,:],dtype) #[HxW]
	diff_img0_yyy = derivative_yyy_greylevel(u[0,:,:],dtype) #[HxW]

	diff_img1_xxx = derivative_xxx_greylevel(u[1,:,:],dtype) #[HxW]
	diff_img1_xxy = derivative_xxy_greylevel(u[1,:,:],dtype) #[HxW]
	diff_img1_xyy = derivative_xyy_greylevel(u[1,:,:],dtype) #[HxW]
	diff_img1_yyy = derivative_yyy_greylevel(u[1,:,:],dtype) #[HxW]

	diff_img2_xxx = derivative_xxx_greylevel(u[2,:,:],dtype) #[HxW]
	diff_img2_xxy = derivative_xxy_greylevel(u[2,:,:],dtype) #[HxW]
	diff_img2_xyy = derivative_xyy_greylevel(u[2,:,:],dtype) #[HxW]
	diff_img2_yyy = derivative_yyy_greylevel(u[2,:,:],dtype) #[HxW]


	terms0_xxx = diff_img0_xxx \
				- 3.*(torch.mul(Gamma1_11,diff_img0_xx)+torch.mul(Gamma2_11,diff_img0_xy)) \
				+ 2.*(torch.mul(torch.square(Gamma1_11),diff_img0_x) + torch.mul(Gamma2_11,torch.mul(Gamma1_12,diff_img0_x)) + torch.mul(Gamma1_11,torch.mul(Gamma2_11,diff_img0_y)) + torch.mul(Gamma2_11,torch.mul(Gamma2_12,diff_img0_y))) \
				- torch.mul(derivative_central_x_greylevel(Gamma1_11,dtype),diff_img0_x) - torch.mul(derivative_central_x_greylevel(Gamma2_11,dtype),diff_img0_y)

	terms1_xxx = diff_img1_xxx \
				- 3.*(torch.mul(Gamma1_11,diff_img1_xx)+torch.mul(Gamma2_11,diff_img1_xy)) \
				+ 2.*(torch.mul(torch.square(Gamma1_11),diff_img1_x) + torch.mul(Gamma2_11,torch.mul(Gamma1_12,diff_img1_x)) + torch.mul(Gamma1_11,torch.mul(Gamma2_11,diff_img1_y)) + torch.mul(Gamma2_11,torch.mul(Gamma2_12,diff_img1_y))) \
				- torch.mul(derivative_central_x_greylevel(Gamma1_11,dtype),diff_img1_x) - torch.mul(derivative_central_x_greylevel(Gamma2_11,dtype),diff_img1_y)

	terms2_xxx = diff_img2_xxx \
				- 3.*(torch.mul(Gamma1_11,diff_img2_xx)+torch.mul(Gamma2_11,diff_img2_xy)) \
				+ 2.*(torch.mul(torch.square(Gamma1_11),diff_img2_x) + torch.mul(Gamma2_11,torch.mul(Gamma1_12,diff_img2_x)) + torch.mul(Gamma1_11,torch.mul(Gamma2_11,diff_img2_y)) + torch.mul(Gamma2_11,torch.mul(Gamma2_12,diff_img2_y))) \
				- torch.mul(derivative_central_x_greylevel(Gamma1_11,dtype),diff_img2_x) - torch.mul(derivative_central_x_greylevel(Gamma2_11,dtype),diff_img2_y)



	terms0_yyy = diff_img0_yyy  \
				- 3.*(torch.mul(Gamma1_22,diff_img0_xy)+torch.mul(Gamma2_22,diff_img0_yy)) \
				+ 2.*(torch.mul(torch.square(Gamma2_22),diff_img0_y) + torch.mul(Gamma1_22,torch.mul(Gamma1_12,diff_img0_x)) + torch.mul(Gamma2_22,torch.mul(Gamma1_22,diff_img0_x)) + torch.mul(Gamma1_22,torch.mul(Gamma2_12,diff_img0_y))) \
				- torch.mul(derivative_central_y_greylevel(Gamma1_22,dtype),diff_img0_x) - torch.mul(derivative_central_y_greylevel(Gamma2_22,dtype),diff_img0_y) 

	terms1_yyy = diff_img1_yyy  \
				- 3.*(torch.mul(Gamma1_22,diff_img1_xy)+torch.mul(Gamma2_22,diff_img1_yy)) \
				+ 2.*(torch.mul(torch.square(Gamma2_22),diff_img1_y) + torch.mul(Gamma1_22,torch.mul(Gamma1_12,diff_img1_x)) + torch.mul(Gamma2_22,torch.mul(Gamma1_22,diff_img1_x)) + torch.mul(Gamma1_22,torch.mul(Gamma2_12,diff_img1_y))) \
				- torch.mul(derivative_central_y_greylevel(Gamma1_22,dtype),diff_img1_x) - torch.mul(derivative_central_y_greylevel(Gamma2_22,dtype),diff_img1_y) 

	terms2_yyy = diff_img2_yyy  \
				- 3.*(torch.mul(Gamma1_22,diff_img2_xy)+torch.mul(Gamma2_22,diff_img2_yy)) \
				+ 2.*(torch.mul(torch.square(Gamma2_22),diff_img2_y) + torch.mul(Gamma1_22,torch.mul(Gamma1_12,diff_img2_x)) + torch.mul(Gamma2_22,torch.mul(Gamma1_22,diff_img2_x)) + torch.mul(Gamma1_22,torch.mul(Gamma2_12,diff_img2_y))) \
				- torch.mul(derivative_central_y_greylevel(Gamma1_22,dtype),diff_img2_x) - torch.mul(derivative_central_y_greylevel(Gamma2_22,dtype),diff_img2_y) 
				
	

	terms0_xxy = diff_img0_xxy \
				- 2.*(torch.mul(Gamma1_12,diff_img0_xx) + torch.mul(Gamma2_12,diff_img0_xy)) - (torch.mul(Gamma1_11,diff_img0_xy) + torch.mul(Gamma2_11,diff_img0_yy)) \
				+ 2.*(torch.mul(Gamma1_11,torch.mul(Gamma1_12,diff_img0_x))) + torch.mul(Gamma1_11,torch.mul(Gamma2_12,diff_img0_y)) + torch.mul(Gamma2_11,torch.mul(Gamma1_22,diff_img0_x))+torch.mul(Gamma2_12,torch.mul(Gamma1_12,diff_img0_x)) \
				+ torch.mul(Gamma2_11,torch.mul(Gamma2_22,diff_img0_y)) + torch.mul(torch.square(Gamma2_12),diff_img0_y) + torch.mul(Gamma1_12,torch.mul(Gamma2_11,diff_img0_y)) \
				- torch.mul(derivative_central_x_greylevel(Gamma1_12,dtype),diff_img0_x) - torch.mul(derivative_central_x_greylevel(Gamma2_12,dtype),diff_img0_y)  	
				
	terms1_xxy = diff_img1_xxy \
				- 2.*(torch.mul(Gamma1_12,diff_img1_xx) + torch.mul(Gamma2_12,diff_img1_xy)) - (torch.mul(Gamma1_11,diff_img1_xy) + torch.mul(Gamma2_11,diff_img1_yy)) \
				+ 2.*(torch.mul(Gamma1_11,torch.mul(Gamma1_12,diff_img1_x))) + torch.mul(Gamma1_11,torch.mul(Gamma2_12,diff_img1_y)) + torch.mul(Gamma2_11,torch.mul(Gamma1_22,diff_img1_x))+torch.mul(Gamma2_12,torch.mul(Gamma1_12,diff_img1_x)) \
				+ torch.mul(Gamma2_11,torch.mul(Gamma2_22,diff_img1_y)) + torch.mul(torch.square(Gamma2_12),diff_img1_y) + torch.mul(Gamma1_12,torch.mul(Gamma2_11,diff_img1_y)) \
				- torch.mul(derivative_central_x_greylevel(Gamma1_12,dtype),diff_img1_x) - torch.mul(derivative_central_x_greylevel(Gamma2_12,dtype),diff_img1_y)  	
	
	terms2_xxy = diff_img2_xxy \
				- 2.*(torch.mul(Gamma1_12,diff_img2_xx) + torch.mul(Gamma2_12,diff_img2_xy)) - (torch.mul(Gamma1_11,diff_img2_xy) + torch.mul(Gamma2_11,diff_img2_yy)) \
				+ 2.*(torch.mul(Gamma1_11,torch.mul(Gamma1_12,diff_img2_x))) + torch.mul(Gamma1_11,torch.mul(Gamma2_12,diff_img2_y)) + torch.mul(Gamma2_11,torch.mul(Gamma1_22,diff_img2_x))+torch.mul(Gamma2_12,torch.mul(Gamma1_12,diff_img2_x)) \
				+ torch.mul(Gamma2_11,torch.mul(Gamma2_22,diff_img2_y)) + torch.mul(torch.square(Gamma2_12),diff_img2_y) + torch.mul(Gamma1_12,torch.mul(Gamma2_11,diff_img2_y)) \
				- torch.mul(derivative_central_x_greylevel(Gamma1_12,dtype),diff_img2_x) - torch.mul(derivative_central_x_greylevel(Gamma2_12,dtype),diff_img2_y)  	
	


	terms0_yxx = diff_img0_xxy \
				-2.*( torch.mul(Gamma1_12,diff_img0_xx) + torch.mul(Gamma2_12,diff_img0_xy)) - (torch.mul(Gamma1_11,diff_img0_xy) + torch.mul(Gamma2_11,diff_img0_yy)) \
				+ 2.*torch.mul(Gamma1_11,torch.mul(Gamma1_12,diff_img0_x) + torch.mul(Gamma2_11,torch.mul(Gamma1_12,diff_img0_y)) + torch.mul(Gamma2_21,torch.mul(Gamma1_12,diff_img0_x)) + torch.mul(torch.square(Gamma2_12),diff_img0_y)) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_11,dtype),diff_img0_x) - torch.mul(derivative_central_y_greylevel(Gamma2_11,dtype),diff_img0_y)	

	terms1_yxx = diff_img1_xxy \
				-2.*( torch.mul(Gamma1_12,diff_img1_xx) + torch.mul(Gamma2_12,diff_img1_xy)) - (torch.mul(Gamma1_11,diff_img1_xy) + torch.mul(Gamma2_11,diff_img1_yy)) \
				+ 2.*torch.mul(Gamma1_11,torch.mul(Gamma1_12,diff_img1_x) + torch.mul(Gamma2_11,torch.mul(Gamma1_12,diff_img1_y)) + torch.mul(Gamma2_21,torch.mul(Gamma1_12,diff_img1_x)) + torch.mul(torch.square(Gamma2_12),diff_img1_y)) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_11,dtype),diff_img1_x) - torch.mul(derivative_central_y_greylevel(Gamma2_11,dtype),diff_img1_y)	

	terms2_yxx = diff_img2_xxy \
				-2.*( torch.mul(Gamma1_12,diff_img2_xx) + torch.mul(Gamma2_12,diff_img2_xy)) - (torch.mul(Gamma1_11,diff_img2_xy) + torch.mul(Gamma2_11,diff_img2_yy)) \
				+ 2.*torch.mul(Gamma1_11,torch.mul(Gamma1_12,diff_img2_x) + torch.mul(Gamma2_11,torch.mul(Gamma1_12,diff_img2_y)) + torch.mul(Gamma2_21,torch.mul(Gamma1_12,diff_img2_x)) + torch.mul(torch.square(Gamma2_12),diff_img2_y)) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_11,dtype),diff_img2_x) - torch.mul(derivative_central_y_greylevel(Gamma2_11,dtype),diff_img2_y)	



	terms0_xyx =	diff_img0_xxy \
				-2.*( torch.mul(Gamma2_12,diff_img0_xy) + torch.mul(Gamma1_12,diff_img0_xx)) - (torch.mul(Gamma1_11,diff_img0_xy)+torch.mul(Gamma2_11,diff_img0_yy)) \
				+ 2.*torch.mul(Gamma1_11,torch.mul(Gamma1_12,diff_img0_x)) + torch.mul(Gamma2_12,torch.mul(Gamma1_12,diff_img0_x)) + torch.mul(Gamma2_11,torch.mul(Gamma1_22,diff_img0_x))  \
				+ torch.mul(Gamma1_12,torch.mul(Gamma2_11,diff_img0_y)) + torch.mul(Gamma1_11,torch.mul(Gamma2_21,diff_img0_y)) + torch.mul(Gamma2_11,torch.mul(Gamma2_22,diff_img0_y)) + torch.mul(torch.square(Gamma2_12),diff_img0_y) \
				-  torch.mul(derivative_central_x_greylevel(Gamma1_21,dtype),diff_img0_x) - torch.mul(derivative_central_x_greylevel(Gamma2_21,dtype),diff_img0_y)

	terms1_xyx =	diff_img1_xxy \
				-2.*( torch.mul(Gamma2_12,diff_img1_xy) + torch.mul(Gamma1_12,diff_img1_xx)) - (torch.mul(Gamma1_11,diff_img1_xy)+torch.mul(Gamma2_11,diff_img1_yy)) \
				+ 2.*torch.mul(Gamma1_11,torch.mul(Gamma1_12,diff_img1_x)) + torch.mul(Gamma2_12,torch.mul(Gamma1_12,diff_img1_x)) + torch.mul(Gamma2_11,torch.mul(Gamma1_22,diff_img1_x))  \
				+ torch.mul(Gamma1_12,torch.mul(Gamma2_11,diff_img1_y)) + torch.mul(Gamma1_11,torch.mul(Gamma2_21,diff_img1_y)) + torch.mul(Gamma2_11,torch.mul(Gamma2_22,diff_img1_y)) + torch.mul(torch.square(Gamma2_12),diff_img1_y) \
				-  torch.mul(derivative_central_x_greylevel(Gamma1_21,dtype),diff_img1_x) - torch.mul(derivative_central_x_greylevel(Gamma2_21,dtype),diff_img1_y)

	terms2_xyx =	diff_img2_xxy \
				-2.*( torch.mul(Gamma2_12,diff_img2_xy) + torch.mul(Gamma1_12,diff_img2_xx)) - (torch.mul(Gamma1_11,diff_img2_xy)+torch.mul(Gamma2_11,diff_img2_yy)) \
				+ 2.*torch.mul(Gamma1_11,torch.mul(Gamma1_12,diff_img2_x)) + torch.mul(Gamma2_12,torch.mul(Gamma1_12,diff_img2_x)) + torch.mul(Gamma2_11,torch.mul(Gamma1_22,diff_img2_x))  \
				+ torch.mul(Gamma1_12,torch.mul(Gamma2_11,diff_img2_y)) + torch.mul(Gamma1_11,torch.mul(Gamma2_21,diff_img2_y)) + torch.mul(Gamma2_11,torch.mul(Gamma2_22,diff_img2_y)) + torch.mul(torch.square(Gamma2_12),diff_img2_y) \
				-  torch.mul(derivative_central_x_greylevel(Gamma1_21,dtype),diff_img2_x) - torch.mul(derivative_central_x_greylevel(Gamma2_21,dtype),diff_img2_y)
	


	terms0_yyx = diff_img0_xyy \
				- 2.*(torch.mul(Gamma1_21,diff_img0_xy) + torch.mul(Gamma2_21,diff_img0_yy)) - (torch.mul(Gamma1_22,diff_img0_xx) + torch.mul(Gamma2_22,diff_img0_xy)) \
				+ 2.*(torch.mul(Gamma2_22,torch.mul(Gamma2_21,diff_img0_y))) + torch.mul(Gamma1_22,torch.mul(Gamma1_11,diff_img0_x)) + torch.mul(Gamma1_22,torch.mul(Gamma2_11,diff_img0_y)) + torch.mul(torch.square(Gamma1_21),diff_img0_x) \
				+ torch.mul(Gamma2_22,torch.mul(Gamma1_21,diff_img0_x)) + torch.mul(Gamma1_21,torch.mul(Gamma2_21,diff_img0_y)) + torch.mul(Gamma2_21,torch.mul(Gamma1_22,diff_img0_x)) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_21,dtype),diff_img0_x) - torch.mul(derivative_central_y_greylevel(Gamma2_21,dtype),diff_img0_y)

	terms1_yyx = diff_img1_xyy \
				- 2.*(torch.mul(Gamma1_21,diff_img1_xy) + torch.mul(Gamma2_21,diff_img1_yy)) - (torch.mul(Gamma1_22,diff_img1_xx) + torch.mul(Gamma2_22,diff_img1_xy)) \
				+ 2.*(torch.mul(Gamma2_22,torch.mul(Gamma2_21,diff_img1_y))) + torch.mul(Gamma1_22,torch.mul(Gamma1_11,diff_img1_x)) + torch.mul(Gamma1_22,torch.mul(Gamma2_11,diff_img1_y)) + torch.mul(torch.square(Gamma1_21),diff_img1_x) \
				+ torch.mul(Gamma2_22,torch.mul(Gamma1_21,diff_img1_x)) + torch.mul(Gamma1_21,torch.mul(Gamma2_21,diff_img1_y)) + torch.mul(Gamma2_21,torch.mul(Gamma1_22,diff_img1_x)) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_21,dtype),diff_img1_x) - torch.mul(derivative_central_y_greylevel(Gamma2_21,dtype),diff_img1_y)
				
	terms2_yyx = diff_img2_xyy \
				- 2.*(torch.mul(Gamma1_21,diff_img2_xy) + torch.mul(Gamma2_21,diff_img2_yy)) - (torch.mul(Gamma1_22,diff_img2_xx) + torch.mul(Gamma2_22,diff_img2_xy)) \
				+ 2.*(torch.mul(Gamma2_22,torch.mul(Gamma2_21,diff_img2_y))) + torch.mul(Gamma1_22,torch.mul(Gamma1_11,diff_img2_x)) + torch.mul(Gamma1_22,torch.mul(Gamma2_11,diff_img2_y)) + torch.mul(torch.square(Gamma1_21),diff_img2_x) \
				+ torch.mul(Gamma2_22,torch.mul(Gamma1_21,diff_img2_x)) + torch.mul(Gamma1_21,torch.mul(Gamma2_21,diff_img2_y)) + torch.mul(Gamma2_21,torch.mul(Gamma1_22,diff_img2_x)) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_21,dtype),diff_img2_x) - torch.mul(derivative_central_y_greylevel(Gamma2_21,dtype),diff_img2_y)						

	
	terms0_xyy = diff_img0_xyy \
				- 2.*(torch.mul(Gamma1_12,diff_img0_xy) + torch.mul(Gamma2_12,diff_img0_yy)) - (torch.mul(Gamma2_22,diff_img0_xy) + torch.mul(Gamma1_22,diff_img0_xx)) \
				+ 2.*torch.mul(Gamma1_12,torch.mul(Gamma2_12,diff_img0_y) + torch.mul(Gamma2_12,torch.mul(Gamma1_22,diff_img0_x)) + torch.mul(Gamma2_22,torch.mul(Gamma2_12,diff_img0_y)) + torch.mul(torch.square(Gamma1_12),diff_img0_x)) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_22,dtype),diff_img0_x) - torch.mul(derivative_central_y_greylevel(Gamma2_22,dtype),diff_img0_y)

	terms1_xyy = diff_img1_xyy \
				- 2.*(torch.mul(Gamma1_12,diff_img1_xy) + torch.mul(Gamma2_12,diff_img1_yy)) - (torch.mul(Gamma2_22,diff_img1_xy) + torch.mul(Gamma1_22,diff_img1_xx)) \
				+ 2.*torch.mul(Gamma1_12,torch.mul(Gamma2_12,diff_img1_y) + torch.mul(Gamma2_12,torch.mul(Gamma1_22,diff_img1_x)) + torch.mul(Gamma2_22,torch.mul(Gamma2_12,diff_img1_y)) + torch.mul(torch.square(Gamma1_12),diff_img1_x)) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_22,dtype),diff_img1_x) - torch.mul(derivative_central_y_greylevel(Gamma2_22,dtype),diff_img1_y)

	terms2_xyy = diff_img2_xyy \
				- 2.*(torch.mul(Gamma1_12,diff_img2_xy) + torch.mul(Gamma2_12,diff_img2_yy)) - (torch.mul(Gamma2_22,diff_img2_xy) + torch.mul(Gamma1_22,diff_img2_xx)) \
				+ 2.*torch.mul(Gamma1_12,torch.mul(Gamma2_12,diff_img2_y) + torch.mul(Gamma2_12,torch.mul(Gamma1_22,diff_img2_x)) + torch.mul(Gamma2_22,torch.mul(Gamma2_12,diff_img2_y)) + torch.mul(torch.square(Gamma1_12),diff_img2_x)) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_22,dtype),diff_img2_x) - torch.mul(derivative_central_y_greylevel(Gamma2_22,dtype),diff_img2_y)


	terms0_yxy = diff_img0_xyy \
				-2.*( torch.mul(Gamma1_12,diff_img0_xy) + torch.mul(Gamma2_12,diff_img0_yy)) - (torch.mul(Gamma2_22,diff_img0_xy)+torch.mul(Gamma1_22,diff_img0_xx)) \
				+ 2.*torch.mul(Gamma2_22,torch.mul(Gamma2_12,diff_img0_y)) + torch.mul(Gamma1_21,torch.mul(Gamma2_21,diff_img0_y)) + torch.mul(Gamma2_21,torch.mul(Gamma1_22,diff_img0_x))  \
				+ torch.mul(Gamma1_22,torch.mul(Gamma1_11,diff_img0_x)) + torch.mul(Gamma1_22,torch.mul(Gamma2_11,diff_img0_y)) + torch.mul(Gamma2_22,torch.mul(Gamma1_12,diff_img0_x)) + torch.mul(torch.square(Gamma1_21),diff_img0_x) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_12,dtype),diff_img0_x) - torch.mul(derivative_central_y_greylevel(Gamma2_12,dtype),diff_img0_y)

	terms1_yxy = diff_img1_xyy \
				-2.*( torch.mul(Gamma1_12,diff_img1_xy) + torch.mul(Gamma2_12,diff_img1_yy)) - (torch.mul(Gamma2_22,diff_img1_xy)+torch.mul(Gamma1_22,diff_img1_xx)) \
				+ 2.*torch.mul(Gamma2_22,torch.mul(Gamma2_12,diff_img1_y)) + torch.mul(Gamma1_21,torch.mul(Gamma2_21,diff_img1_y)) + torch.mul(Gamma2_21,torch.mul(Gamma1_22,diff_img1_x))  \
				+ torch.mul(Gamma1_22,torch.mul(Gamma1_11,diff_img1_x)) + torch.mul(Gamma1_22,torch.mul(Gamma2_11,diff_img1_y)) + torch.mul(Gamma2_22,torch.mul(Gamma1_12,diff_img1_x)) + torch.mul(torch.square(Gamma1_21),diff_img1_x) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_12,dtype),diff_img1_x) - torch.mul(derivative_central_y_greylevel(Gamma2_12,dtype),diff_img1_y)
				
	terms2_yxy =	diff_img2_xyy \
				-2.*( torch.mul(Gamma1_12,diff_img2_xy) + torch.mul(Gamma2_12,diff_img2_yy)) - (torch.mul(Gamma2_22,diff_img2_xy)+torch.mul(Gamma1_22,diff_img2_xx)) \
				+ 2.*torch.mul(Gamma2_22,torch.mul(Gamma2_12,diff_img2_y)) + torch.mul(Gamma1_21,torch.mul(Gamma2_21,diff_img2_y)) + torch.mul(Gamma2_21,torch.mul(Gamma1_22,diff_img2_x))  \
				+ torch.mul(Gamma1_22,torch.mul(Gamma1_11,diff_img2_x)) + torch.mul(Gamma1_22,torch.mul(Gamma2_11,diff_img2_y)) + torch.mul(Gamma2_22,torch.mul(Gamma1_12,diff_img2_x)) + torch.mul(torch.square(Gamma1_21),diff_img2_x) \
				-  torch.mul(derivative_central_y_greylevel(Gamma1_12,dtype),diff_img2_x) - torch.mul(derivative_central_y_greylevel(Gamma2_12,dtype),diff_img2_y)						


	norm_third_order_covariant_derivative_squared = torch.mul(torch.pow(invg11,3),torch.square(terms0_xxx)+torch.square(terms1_xxx)+torch.square(terms2_xxx)) \
		+ torch.mul(torch.pow(invg22,3),torch.square(terms0_yyy)+torch.square(terms1_yyy)+torch.square(terms2_yyy)) \
		+ 2.*(torch.mul(torch.pow(invg12,3),torch.mul(terms0_xxx,terms0_yyy) + torch.mul(terms1_xxx,terms1_yyy) + torch.mul(terms2_xxx,terms2_yyy) + torch.mul(terms0_xxy,terms0_yyx) + torch.mul(terms1_xxy,terms1_yyx) + torch.mul(terms2_xxy,terms2_yyx) + torch.mul(terms0_xyx,terms0_yxy) + torch.mul(terms1_xyx,terms1_yxy) + torch.mul(terms2_xyx,terms2_yxy) +  torch.mul(terms0_xyy,terms0_yxx) + torch.mul(terms1_xyy,terms1_yxx) + torch.mul(terms2_xyy,terms2_yxx))) \
		+ 2.*(torch.mul(torch.mul(invg11,torch.mul(invg12,invg22)),torch.mul(terms0_xyx,terms0_xyy) + torch.mul(terms1_xyx,terms1_xyy) + torch.mul(terms2_xyx,terms2_xyy) + torch.mul(terms0_yxx,terms0_yxy) + torch.mul(terms1_yxx,terms1_yxy) + torch.mul(terms2_yxx,terms2_yxy)  + torch.mul(terms0_xxy,terms0_xyy) + torch.mul(terms1_xxy,terms1_xyy) + torch.mul(terms2_xxy,terms2_xyy) + torch.mul(terms0_yxx,terms0_yyx) + torch.mul(terms1_yxx,terms1_yyx) + torch.mul(terms2_yxx,terms2_yyx) + torch.mul(terms0_xxy,terms0_yxy) + torch.mul(terms1_xxy,terms1_yxy) + torch.mul(terms2_xxy,terms2_yxy) + torch.mul(terms0_xyx,terms0_yyx) + torch.mul(terms1_xyx,terms1_yyx) + torch.mul(terms2_xyx,terms2_yyx))) \
		+ 2.*(torch.mul(torch.mul(torch.square(invg11),invg12),torch.mul(terms0_xxx,terms0_xxy) + torch.mul(terms1_xxx,terms1_xxy) + torch.mul(terms2_xxx,terms2_xxy) + torch.mul(terms0_xxx,terms0_xyx) + torch.mul(terms1_xxx,terms1_xyx) + torch.mul(terms2_xxx,terms2_xyx) + torch.mul(terms0_xxx,terms0_yxx) + torch.mul(terms1_xxx,terms1_yxx) + torch.mul(terms2_xxx,terms2_yxx))) \
		+ 2.*(torch.mul(torch.mul(torch.square(invg22),invg12),torch.mul(terms0_yyx,terms0_yyy) + torch.mul(terms1_yyx,terms1_yyy) + torch.mul(terms2_yyx,terms2_yyy) + torch.mul(terms0_yxy,terms0_yyy) + torch.mul(terms1_yxy,terms1_yyy) + torch.mul(terms2_yxy,terms2_yyy) + torch.mul(terms0_xyy,terms0_yyy) + torch.mul(terms1_xyy,terms1_yyy) + torch.mul(terms2_xyy,terms2_yyy))) \
		+ 2.*(torch.mul(torch.mul(torch.square(invg12),invg11),torch.mul(terms0_xxx,terms0_xyy) + torch.mul(terms1_xxx,terms1_xyy) + torch.mul(terms2_xxx,terms2_xyy) + torch.mul(terms0_xxy,terms0_xyx) + torch.mul(terms1_xxy,terms1_xyx) + torch.mul(terms2_xxy,terms2_xyx) + torch.mul(terms0_xxx,terms0_yxy) + torch.mul(terms1_xxx,terms1_yxy) + torch.mul(terms2_xxx,terms2_yxy) + torch.mul(terms0_xxy,terms0_yxx) + torch.mul(terms1_xxy,terms1_yxx) + torch.mul(terms2_xxy,terms2_yxx) + torch.mul(terms0_xxx,terms0_yyx) + torch.mul(terms1_xxx,terms1_yyx) + torch.mul(terms2_xxx,terms2_yyx) + torch.mul(terms0_xyx,terms0_yxx) + torch.mul(terms1_xyx,terms1_yxx) + torch.mul(terms2_xyx,terms2_yxx))) \
		+ 2.*(torch.mul(torch.mul(torch.square(invg12),invg22),torch.mul(terms0_yxx,terms0_yyy) + torch.mul(terms1_yxx,terms1_yyy) + torch.mul(terms2_yxx,terms2_yyy) + torch.mul(terms0_yxy,terms0_yyx) + torch.mul(terms1_yxy,terms1_yyx) + torch.mul(terms2_yxy,terms2_yyx) + torch.mul(terms0_xyx,terms0_yyy) + torch.mul(terms1_xyx,terms1_yyy) + torch.mul(terms2_xyx,terms2_yyy)	+ torch.mul(terms0_xyy,terms0_yyx) + torch.mul(terms1_xyy,terms1_yyx) + torch.mul(terms2_xyy,terms2_yyx) + torch.mul(terms0_xxy,terms0_yyy) + torch.mul(terms1_xxy,terms1_yyy) + torch.mul(terms2_xxy,terms2_yyy) + torch.mul(terms0_xyy,terms0_yxy) + torch.mul(terms1_xyy,terms1_yxy) + torch.mul(terms2_xyy,terms2_yxy))) \
		+ torch.mul(torch.mul(torch.square(invg11),invg22),torch.square(terms0_xxy) + torch.square(terms1_xxy) + torch.square(terms2_xxy) + torch.square(terms0_xyx) + torch.square(terms1_xyx) + torch.square(terms2_xyx) + torch.square(terms0_yxx) + torch.square(terms1_yxx) + torch.square(terms2_yxx)) + torch.mul(torch.mul(torch.square(invg22),invg11),torch.square(terms0_xyy) + torch.square(terms1_xyy) + torch.square(terms2_xyy) + torch.square(terms0_yxy) + torch.square(terms1_yxy) + torch.square(terms2_yxy) + torch.square(terms0_yyx) + torch.square(terms1_yyx) + torch.square(terms2_yyx))

		 		
	norm_third_order_covariant_derivative = torch.sqrt(epsilon*torch.ones(M,N).type(dtype)+norm_third_order_covariant_derivative_squared)

	return norm_third_order_covariant_derivative






