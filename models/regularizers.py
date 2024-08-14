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

	'''u_plus2_plus1 = u[2:,1:] #[H-2xW-1]
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

	u_xxy = (u_plus2_plus1-2.*u_0_plus1+u_minus2_plus1-u_plus2_minus1+2.*u_0_minus1-u_minus2_minus1)/8.'''

	u_0_plus1 = u[:,1:] #[HxW-1]
	u_0_plus1 = torch.cat((u_0_plus1,torch.zeros(M,1).type(dtype)),axis=1) #[HxW]

	u_0_minus1 = u[:,:-1] #[HxW-1]
	u_0_minus1 = torch.cat((torch.zeros(M,1).type(dtype),u_0_minus1),axis=1) #[HxW]

	u_plus1_plus1 = u[1:,1:] #[H-1xW-1]
	u_plus1_plus1 = torch.cat((u_plus1_plus1,torch.zeros(1,N-1).type(dtype)),axis=0) #[HxW-1]
	u_plus1_plus1 = torch.cat((u_plus1_plus1,torch.zeros(M,1).type(dtype)),axis=1)   #[HxW]

	u_minus1_plus1 = u[:-1,1:] #[H-1xW-1]
	u_minus1_plus1 = torch.cat((torch.zeros(1,N-1).type(dtype),u_minus1_plus1),axis=0) #[HxW-1]
	u_minus1_plus1 = torch.cat((u_minus1_plus1,torch.zeros(M,1).type(dtype)),axis=1) #[HxW]

	u_plus1_minus1 = u[1:,:-1] #[H-1xW-1]
	u_plus1_minus1 = torch.cat((u_plus1_minus1,torch.zeros(1,N-1).type(dtype)),axis=0) #[HxW-1]
	u_plus1_minus1 = torch.cat((torch.zeros(M,1).type(dtype),u_plus1_minus1),axis=1) #[HxW]

	u_minus1_minus1 = u[:-1,:-1] #[H-1xW-1]
	u_minus1_minus1 = torch.cat((torch.zeros(1,N-1).type(dtype),u_minus1_minus1),axis=0) #[HxW-1]
	u_minus1_minus1 = torch.cat((torch.zeros(M,1).type(dtype),u_minus1_minus1),axis=1) #[HxW]

	u_xxy = (u_minus1_plus1-u_minus1_minus1+u_plus1_plus1-u_plus1_minus1+2.*u_0_minus1-2.*u_0_plus1)/2.

	return u_xxy


def derivative_xyy_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	'''u_plus1_plus2 = u[1:,2:] #[H-1xW-2]
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



	u_xyy = (u_plus1_plus2-2.*u_plus1_0+u_plus1_minus2-u_minus1_plus2+2.*u_minus1_0-u_minus1_minus2)/8.'''

	u_plus1_0 = u[1:,:] #[H-1xW]
	u_plus1_0 = torch.cat((u_plus1_0,torch.zeros(1,N).type(dtype)),axis=0) #[HxW]

	u_minus1_0 = u[:-1,:] #[H-1xW]
	u_minus1_0 = torch.cat((torch.zeros(1,N).type(dtype),u_minus1_0),axis=0) #[HxW]

	u_plus1_plus1 = u[1:,1:] #[H-1xW-1]
	u_plus1_plus1 = torch.cat((u_plus1_plus1,torch.zeros(1,N-1).type(dtype)),axis=0) #[HxW-1]
	u_plus1_plus1 = torch.cat((u_plus1_plus1,torch.zeros(M,1).type(dtype)),axis=1)   #[HxW]

	u_minus1_plus1 = u[:-1,1:] #[H-1xW-1]
	u_minus1_plus1 = torch.cat((torch.zeros(1,N-1).type(dtype),u_minus1_plus1),axis=0) #[HxW-1]
	u_minus1_plus1 = torch.cat((u_minus1_plus1,torch.zeros(M,1).type(dtype)),axis=1) #[HxW]

	u_plus1_minus1 = u[1:,:-1] #[H-1xW-1]
	u_plus1_minus1 = torch.cat((u_plus1_minus1,torch.zeros(1,N-1).type(dtype)),axis=0) #[HxW-1]
	u_plus1_minus1 = torch.cat((torch.zeros(M,1).type(dtype),u_plus1_minus1),axis=1) #[HxW]

	u_minus1_minus1 = u[:-1,:-1] #[H-1xW-1]
	u_minus1_minus1 = torch.cat((torch.zeros(1,N-1).type(dtype),u_minus1_minus1),axis=0) #[HxW-1]
	u_minus1_minus1 = torch.cat((torch.zeros(M,1).type(dtype),u_minus1_minus1),axis=1) #[HxW]



	u_xyy = (u_plus1_minus1-u_minus1_minus1+u_plus1_plus1-u_minus1_plus1+2.*u_minus1_0-2.*u_plus1_0)/2.

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

def covariant_derivative_first_order(u,omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,dtype):

	u1 = torch.narrow(u,1,0,1)       # [1x1xHxW]
	u1 = torch.squeeze(u1)   		 # [HxW]
	u1 = torch.unsqueeze(u1,dim=0)   # [1xHxW]
	u1tilde = u1.repeat(2,1,1)   	 # [2xHxW]
	u2 = torch.narrow(u,1,1,1)       # [1x1xHxW]
	u2 = torch.squeeze(u2)   		 # [HxW]
	u2 = torch.unsqueeze(u2,dim=0)   # [1xHxW]
	u2tilde = u2.repeat(2,1,1)   	 # [2xHxW]
	u3 = torch.narrow(u,1,2,1)       # [1x1xHxW]
	u3 = torch.squeeze(u3)   		 # [HxW]
	u3 = torch.unsqueeze(u3,dim=0)   # [1xHxW]
	u3tilde = u3.repeat(2,1,1)       # [2xHxW]

	zero_order_term1 = torch.mul(omegaE_11,u1tilde) + torch.mul(omegaE_12,u2tilde) + torch.mul(omegaE_13,u3tilde) #[2xHxW]
	zero_order_term1 = torch.unsqueeze(zero_order_term1,dim=0)														   #[1x2xHxW]
	zero_order_term2 = torch.mul(omegaE_21,u1tilde) + torch.mul(omegaE_22,u2tilde) + torch.mul(omegaE_23,u3tilde) #[2xHxW]
	zero_order_term2 = torch.unsqueeze(zero_order_term2,dim=0)														   #[1x2xHxW]
	zero_order_term3 = torch.mul(omegaE_31,u1tilde) + torch.mul(omegaE_32,u2tilde) + torch.mul(omegaE_33,u3tilde) #[2xHxW]
	zero_order_term3 = torch.unsqueeze(zero_order_term3,dim=0)														   #[1x2xHxW]

	zero_order_term  = torch.cat((zero_order_term1,zero_order_term2,zero_order_term3),dim=0)                           #[3x2xHxW]

	diff_u1_x = derivative_central_x_greylevel(torch.squeeze(u1),dtype)     #[HxW]
	diff_u1_x = torch.unsqueeze(diff_u1_x,dim=0)							#[1xHxW]
	diff_u1_y = derivative_central_y_greylevel(torch.squeeze(u1),dtype)     #[HxW]
	diff_u1_y = torch.unsqueeze(diff_u1_y,dim=0)							#[1xHxW]
	diff_u1 = torch.cat((diff_u1_x,diff_u1_y),dim=0)                        #[2xHxW]
	diff_u1 = torch.unsqueeze(diff_u1,dim=0)								#[1x2xHxW]

	diff_u2_x = derivative_central_x_greylevel(torch.squeeze(u2),dtype)     #[HxW]
	diff_u2_x = torch.unsqueeze(diff_u2_x,dim=0)							#[1xHxW]
	diff_u2_y = derivative_central_y_greylevel(torch.squeeze(u2),dtype)     #[HxW]
	diff_u2_y = torch.unsqueeze(diff_u2_y,dim=0)							#[1xHxW]
	diff_u2 = torch.cat((diff_u2_x,diff_u2_y),dim=0)                        #[2xHxW]
	diff_u2 = torch.unsqueeze(diff_u2,dim=0)								#[1x2xHxW]

	diff_u3_x = derivative_central_x_greylevel(torch.squeeze(u3),dtype)     #[HxW]
	diff_u3_x = torch.unsqueeze(diff_u3_x,dim=0)							#[1xHxW]
	diff_u3_y = derivative_central_y_greylevel(torch.squeeze(u3),dtype)     #[HxW]
	diff_u3_y = torch.unsqueeze(diff_u3_y,dim=0)							#[1xHxW]
	diff_u3 = torch.cat((diff_u3_x,diff_u3_y),dim=0)                        #[2xHxW]
	diff_u3 = torch.unsqueeze(diff_u3,dim=0)								#[1x2xHxW]

	diff_u = torch.cat((diff_u1,diff_u2,diff_u3),dim=0)						#[3x2xHxW]

	cov_dev_u = diff_u + zero_order_term  									#[3x2xHxW]

	return cov_dev_u

def norm_covariant_derivative_first_order(u,epsilon, omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,invg11,invg12,invg22,dtype):	

	cov_dev_u = covariant_derivative_first_order(u,omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,dtype)

	norm_cov_dev_order1_squared = torch.mul(invg11,torch.sum(torch.square(cov_dev_u[:,0,:,:]),dim=0)) + 2.*torch.mul(invg12,torch.sum(torch.mul(cov_dev_u[:,0,:,:],cov_dev_u[:,1,:,:]),dim=0)) + torch.mul(invg22,torch.sum(torch.square(cov_dev_u[:,1,:,:]),dim=0))

	norm_cov_dev_order1 = torch.sqrt(epsilon+norm_cov_dev_order1_squared) #[HxW]

	return norm_cov_dev_order1

def norm_covariant_derivative_first_order_GL3_optimal(u,epsilon, omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,invg11,invg12,invg22,dtype):	

	cov_dev_u = covariant_derivative_first_order_GL3_optimal(u,omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,dtype)

	norm_cov_dev_order1_squared = torch.mul(invg11,torch.sum(torch.square(cov_dev_u[:,0,:,:]),dim=0)) + 2.*torch.mul(invg12,torch.sum(torch.mul(cov_dev_u[:,0,:,:],cov_dev_u[:,1,:,:]),dim=0)) + torch.mul(invg22,torch.sum(torch.square(cov_dev_u[:,1,:,:]),dim=0))

	norm_cov_dev_order1 = torch.sqrt(epsilon+norm_cov_dev_order1_squared) #[HxW]

	return norm_cov_dev_order1	


def covariant_derivative_second_order_general_formula(u,omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,omegaTM_11,omegaTM_12,omegaTM_21,omegaTM_22,dtype):

	cov_dev_u = covariant_derivative_first_order(u,omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,dtype) #[1x3x2xHxW]

	cov_dev_order2_xx_1 = derivative_central_x_greylevel(cov_dev_u[0,0,:,:],dtype) + torch.mul(omegaE_11[0,:,:],cov_dev_u[0,0,:,:]) + torch.mul(omegaE_12[0,:,:],cov_dev_u[1,0,:,:]) + torch.mul(omegaE_13[0,:,:],cov_dev_u[2,0,:,:]) - torch.mul(omegaTM_11[0,:,:],cov_dev_u[0,0,:,:]) - torch.mul(omegaTM_11[1,:,:],cov_dev_u[0,1,:,:]) #[HxW]
	cov_dev_order2_xx_1 = torch.unsqueeze(cov_dev_order2_xx_1,dim=0) #[1xHxW]
	cov_dev_order2_xy_1 = derivative_central_x_greylevel(cov_dev_u[0,1,:,:],dtype) + torch.mul(omegaE_11[0,:,:],cov_dev_u[0,1,:,:]) + torch.mul(omegaE_12[0,:,:],cov_dev_u[1,1,:,:]) + torch.mul(omegaE_13[0,:,:],cov_dev_u[2,1,:,:]) - torch.mul(omegaTM_12[0,:,:],cov_dev_u[0,0,:,:]) - torch.mul(omegaTM_12[1,:,:],cov_dev_u[0,1,:,:]) #[HxW]
	cov_dev_order2_xy_1 = torch.unsqueeze(cov_dev_order2_xy_1,dim=0) #[1xHxW]
	cov_dev_order2_yx_1 = derivative_central_y_greylevel(cov_dev_u[0,0,:,:],dtype) + torch.mul(omegaE_11[1,:,:],cov_dev_u[0,0,:,:]) + torch.mul(omegaE_12[1,:,:],cov_dev_u[1,0,:,:]) + torch.mul(omegaE_13[1,:,:],cov_dev_u[2,0,:,:]) - torch.mul(omegaTM_21[0,:,:],cov_dev_u[0,0,:,:]) - torch.mul(omegaTM_21[1,:,:],cov_dev_u[0,1,:,:]) #[HxW]
	cov_dev_order2_yx_1 = torch.unsqueeze(cov_dev_order2_yx_1,dim=0) #[1xHxW]
	cov_dev_order2_yy_1 = derivative_central_y_greylevel(cov_dev_u[0,1,:,:],dtype) + torch.mul(omegaE_11[1,:,:],cov_dev_u[0,1,:,:]) + torch.mul(omegaE_12[1,:,:],cov_dev_u[1,1,:,:]) + torch.mul(omegaE_13[1,:,:],cov_dev_u[2,1,:,:]) - torch.mul(omegaTM_22[0,:,:],cov_dev_u[0,0,:,:]) - torch.mul(omegaTM_22[1,:,:],cov_dev_u[0,1,:,:]) #[HxW]
	cov_dev_order2_yy_1 = torch.unsqueeze(cov_dev_order2_yy_1,dim=0) #[1xHxW]
	
	cov_dev_order2_1 = torch.cat((cov_dev_order2_xx_1,cov_dev_order2_xy_1,cov_dev_order2_yx_1,cov_dev_order2_yy_1),dim=0) #[4xHxW]
	cov_dev_order2_1 = torch.unsqueeze(cov_dev_order2_1,dim=0) #[1x4xHxW]

	cov_dev_order2_xx_2 = derivative_central_x_greylevel(cov_dev_u[1,0,:,:],dtype) + torch.mul(omegaE_21[0,:,:],cov_dev_u[0,0,:,:]) + torch.mul(omegaE_22[0,:,:],cov_dev_u[1,0,:,:]) + torch.mul(omegaE_23[0,:,:],cov_dev_u[2,0,:,:]) - torch.mul(omegaTM_11[0,:,:],cov_dev_u[1,0,:,:])	- torch.mul(omegaTM_11[1,:,:],cov_dev_u[1,1,:,:])	#[HxW]
	cov_dev_order2_xx_2 = torch.unsqueeze(cov_dev_order2_xx_2,dim=0) #[1xHxW]
	cov_dev_order2_xy_2 = derivative_central_x_greylevel(cov_dev_u[1,1,:,:],dtype) + torch.mul(omegaE_21[0,:,:],cov_dev_u[0,1,:,:]) + torch.mul(omegaE_22[0,:,:],cov_dev_u[1,1,:,:]) + torch.mul(omegaE_23[0,:,:],cov_dev_u[2,1,:,:]) - torch.mul(omegaTM_12[0,:,:],cov_dev_u[1,0,:,:]) - torch.mul(omegaTM_12[1,:,:],cov_dev_u[1,1,:,:]) #[HxW]
	cov_dev_order2_xy_2 = torch.unsqueeze(cov_dev_order2_xy_2,dim=0) #[1xHxW]
	cov_dev_order2_yx_2 = derivative_central_y_greylevel(cov_dev_u[1,0,:,:],dtype) + torch.mul(omegaE_21[1,:,:],cov_dev_u[0,0,:,:]) + torch.mul(omegaE_22[1,:,:],cov_dev_u[1,0,:,:]) + torch.mul(omegaE_23[1,:,:],cov_dev_u[2,0,:,:]) - torch.mul(omegaTM_21[0,:,:],cov_dev_u[1,0,:,:]) - torch.mul(omegaTM_21[1,:,:],cov_dev_u[1,1,:,:]) #[HxW]
	cov_dev_order2_yx_2 = torch.unsqueeze(cov_dev_order2_yx_2,dim=0) #[1xHxW]
	cov_dev_order2_yy_2 = derivative_central_y_greylevel(cov_dev_u[1,1,:,:],dtype) + torch.mul(omegaE_21[1,:,:],cov_dev_u[0,1,:,:]) + torch.mul(omegaE_22[1,:,:],cov_dev_u[1,1,:,:]) + torch.mul(omegaE_23[1,:,:],cov_dev_u[2,1,:,:]) - torch.mul(omegaTM_22[0,:,:],cov_dev_u[1,0,:,:]) - torch.mul(omegaTM_22[1,:,:],cov_dev_u[1,1,:,:]) #[HxW]
	cov_dev_order2_yy_2 = torch.unsqueeze(cov_dev_order2_yy_2,dim=0) #[1xHxW]

	cov_dev_order2_2 = torch.cat((cov_dev_order2_xx_2,cov_dev_order2_xy_2,cov_dev_order2_yx_2,cov_dev_order2_yy_2),dim=0) #[4xHxW]
	cov_dev_order2_2 = torch.unsqueeze(cov_dev_order2_2,dim=0) #[1x4xHxW]

	cov_dev_order2_xx_3 = derivative_central_x_greylevel(cov_dev_u[2,0,:,:],dtype) + torch.mul(omegaE_31[0,:,:],cov_dev_u[0,0,:,:]) + torch.mul(omegaE_32[0,:,:],cov_dev_u[1,0,:,:]) + torch.mul(omegaE_33[0,:,:],cov_dev_u[2,0,:,:]) - torch.mul(omegaTM_11[0,:,:],cov_dev_u[2,0,:,:])	- torch.mul(omegaTM_11[1,:,:],cov_dev_u[2,1,:,:])	#[HxW]
	cov_dev_order2_xx_3 = torch.unsqueeze(cov_dev_order2_xx_3,dim=0) #[1xHxW]
	cov_dev_order2_xy_3 = derivative_central_x_greylevel(cov_dev_u[2,1,:,:],dtype) + torch.mul(omegaE_31[0,:,:],cov_dev_u[0,1,:,:]) + torch.mul(omegaE_32[0,:,:],cov_dev_u[1,1,:,:]) + torch.mul(omegaE_33[0,:,:],cov_dev_u[2,1,:,:]) - torch.mul(omegaTM_12[0,:,:],cov_dev_u[2,0,:,:]) - torch.mul(omegaTM_12[1,:,:],cov_dev_u[2,1,:,:]) #[HxW]
	cov_dev_order2_xy_3 = torch.unsqueeze(cov_dev_order2_xy_3,dim=0) #[1xHxW]
	cov_dev_order2_yx_3 = derivative_central_y_greylevel(cov_dev_u[2,0,:,:],dtype) + torch.mul(omegaE_31[1,:,:],cov_dev_u[0,0,:,:]) + torch.mul(omegaE_32[1,:,:],cov_dev_u[1,0,:,:]) + torch.mul(omegaE_33[1,:,:],cov_dev_u[2,0,:,:]) - torch.mul(omegaTM_21[0,:,:],cov_dev_u[2,0,:,:]) - torch.mul(omegaTM_21[1,:,:],cov_dev_u[2,1,:,:]) #[HxW]
	cov_dev_order2_yx_3 = torch.unsqueeze(cov_dev_order2_yx_3,dim=0) #[1xHxW]
	cov_dev_order2_yy_3 = derivative_central_y_greylevel(cov_dev_u[2,1,:,:],dtype) + torch.mul(omegaE_31[1,:,:],cov_dev_u[0,1,:,:]) + torch.mul(omegaE_32[1,:,:],cov_dev_u[1,1,:,:]) + torch.mul(omegaE_33[1,:,:],cov_dev_u[2,1,:,:]) - torch.mul(omegaTM_22[0,:,:],cov_dev_u[2,0,:,:]) - torch.mul(omegaTM_22[1,:,:],cov_dev_u[2,1,:,:]) #[HxW]
	cov_dev_order2_yy_3 = torch.unsqueeze(cov_dev_order2_yy_3,dim=0) #[1xHxW]

	cov_dev_order2_3 = torch.cat((cov_dev_order2_xx_3,cov_dev_order2_xy_3,cov_dev_order2_yx_3,cov_dev_order2_yy_3),dim=0) #[4xHxW]
	cov_dev_order2_3 = torch.unsqueeze(cov_dev_order2_3,dim=0) #[1x4xHxW]

	cov_dev_order2 = torch.cat((cov_dev_order2_1,cov_dev_order2_2,cov_dev_order2_3),dim=0) #[3x4xHxW]

	return cov_dev_order2 #[3x4xHxW]


def covariant_derivative_second_order_GL3(u,red, green, blue, diff_red_x,diff_green_x,diff_blue_x, diff_red_y,diff_green_y,diff_blue_y,diff_red_xx,diff_red_xy,diff_red_yy,diff_green_xx,diff_green_xy,diff_green_yy,diff_blue_xx,diff_blue_xy,diff_blue_yy, omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,omegaTM_11,omegaTM_12,omegaTM_21,omegaTM_22,dtype):

	cov_dev_u = covariant_derivative_first_order(u,omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,dtype) #[3x2xHxW]


	u1 = torch.narrow(u,1,0,1)       # [1x1xHxW]
	u1 = torch.squeeze(u1)   		 # [HxW]
	u2 = torch.narrow(u,1,1,1)       # [1x1xHxW]
	u2 = torch.squeeze(u2)   		 # [HxW]
	u3 = torch.narrow(u,1,2,1)       # [1x1xHxW]
	u3 = torch.squeeze(u3)   		 # [HxW]

	diff_u1_x = derivative_central_x_greylevel(u1,dtype) #[HxW]
	diff_u1_y = derivative_central_x_greylevel(u1,dtype) #[HxW]

	diff_u2_x = derivative_central_x_greylevel(u2,dtype) #[HxW]
	diff_u2_y = derivative_central_x_greylevel(u2,dtype) #[HxW]

	diff_u3_x = derivative_central_x_greylevel(u3,dtype) #[HxW]
	diff_u3_y = derivative_central_x_greylevel(u3,dtype) #[HxW]

	diff_u1_xx = derivative_xx_greylevel(u1,dtype) #[HxW]
	diff_u1_xy = derivative_xy_greylevel(u1,dtype) #[HxW]
	diff_u1_yy = derivative_yy_greylevel(u1,dtype) #[HxW]

	diff_u2_xx = derivative_xx_greylevel(u2,dtype) #[HxW]
	diff_u2_xy = derivative_xy_greylevel(u2,dtype) #[HxW]
	diff_u2_yy = derivative_yy_greylevel(u2,dtype) #[HxW]

	diff_u3_xx = derivative_xx_greylevel(u3,dtype) #[HxW]
	diff_u3_xy = derivative_xy_greylevel(u3,dtype) #[HxW]
	diff_u3_yy = derivative_yy_greylevel(u3,dtype) #[HxW]

	den = 0.001+torch.square(red)+torch.square(green)+torch.square(blue)
	tmpx = 2.*(torch.div(torch.mul(red,diff_red_x)+torch.mul(green,diff_green_x)+torch.mul(blue,diff_blue_x),torch.square(den)))
	tmpy = 2.*(torch.div(torch.mul(red,diff_red_y)+torch.mul(green,diff_green_y)+torch.mul(blue,diff_blue_y),torch.square(den)))

	cov_dev_order2_xx_1 = diff_u1_xx + torch.mul(-tmpx,torch.mul(omegaE_11[0,:,:],u1)+torch.mul(omegaE_12[0,:,:],u2)+torch.mul(omegaE_13[0,:,:],u3)) \
						- torch.mul((1./den),torch.mul(diff_red_x,torch.mul(diff_red_x,u1))+torch.mul(red,torch.mul(diff_red_xx,u1))+torch.mul(red,torch.mul(diff_red_x,diff_u1_x)) \
						+ torch.mul(diff_green_x,torch.mul(diff_red_x,u2))+torch.mul(green,torch.mul(diff_red_xx,u2))+torch.mul(green,torch.mul(diff_red_x,diff_u2_x))\
						+ torch.mul(diff_blue_x,torch.mul(diff_red_x,u3))+torch.mul(blue,torch.mul(diff_red_xx,u3))+torch.mul(blue,torch.mul(diff_red_x,diff_u3_x))) \
						+ torch.mul(omegaE_11[0,:,:],cov_dev_u[0,0,:,:])+torch.mul(omegaE_12[0,:,:],cov_dev_u[1,0,:,:])+torch.mul(omegaE_13[0,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_xx_1 = torch.unsqueeze(cov_dev_order2_xx_1,dim=0) #[1xHxW]					

	cov_dev_order2_xy_1 = diff_u1_xy + torch.mul(-tmpx,torch.mul(omegaE_11[1,:,:],u1)+torch.mul(omegaE_12[1,:,:],u2)+torch.mul(omegaE_13[1,:,:],u3)) \
						- torch.mul((1./den),torch.mul(diff_red_x,torch.mul(diff_red_y,u1))+torch.mul(red,torch.mul(diff_red_xy,u1))+torch.mul(red,torch.mul(diff_red_y,diff_u1_x)) \
						+ torch.mul(diff_green_x,torch.mul(diff_red_y,u2))+torch.mul(green,torch.mul(diff_red_xy,u2))+torch.mul(green,torch.mul(diff_red_y,diff_u2_x))\
						+ torch.mul(diff_blue_x,torch.mul(diff_red_y,u3))+torch.mul(blue,torch.mul(diff_red_xy,u3))+torch.mul(blue,torch.mul(diff_red_y,diff_u3_x))) \
						+ torch.mul(omegaE_11[0,:,:],cov_dev_u[0,1,:,:])+torch.mul(omegaE_12[0,:,:],cov_dev_u[1,1,:,:])+torch.mul(omegaE_13[0,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_xy_1 = torch.unsqueeze(cov_dev_order2_xy_1,dim=0) #[1xHxW]					

	cov_dev_order2_yx_1 = diff_u1_xy + torch.mul(-tmpy,torch.mul(omegaE_11[0,:,:],u1)+torch.mul(omegaE_12[0,:,:],u2)+torch.mul(omegaE_13[0,:,:],u3)) \
						- torch.mul((1./den),torch.mul(diff_red_y,torch.mul(diff_red_x,u1))+torch.mul(red,torch.mul(diff_red_xy,u1))+torch.mul(red,torch.mul(diff_red_x,diff_u1_y)) \
						+ torch.mul(diff_green_y,torch.mul(diff_red_x,u2))+torch.mul(green,torch.mul(diff_red_xy,u2))+torch.mul(green,torch.mul(diff_red_x,diff_u2_y))\
						+ torch.mul(diff_blue_y,torch.mul(diff_red_x,u3))+torch.mul(blue,torch.mul(diff_red_xy,u3))+torch.mul(blue,torch.mul(diff_red_x,diff_u3_y))) \
						+ torch.mul(omegaE_11[1,:,:],cov_dev_u[0,0,:,:])+torch.mul(omegaE_12[1,:,:],cov_dev_u[1,0,:,:])+torch.mul(omegaE_13[1,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_yx_1 = torch.unsqueeze(cov_dev_order2_yx_1,dim=0) #[1xHxW]

	cov_dev_order2_yy_1 = diff_u1_yy + torch.mul(-tmpy,torch.mul(omegaE_11[1,:,:],u1)+torch.mul(omegaE_12[1,:,:],u2)+torch.mul(omegaE_13[1,:,:],u3)) \
						- torch.mul((1./den),torch.mul(diff_red_y,torch.mul(diff_red_y,u1))+torch.mul(red,torch.mul(diff_red_yy,u1))+torch.mul(red,torch.mul(diff_red_y,diff_u1_y)) \
						+ torch.mul(diff_green_y,torch.mul(diff_red_y,u2))+torch.mul(green,torch.mul(diff_red_yy,u2))+torch.mul(green,torch.mul(diff_red_y,diff_u2_y))\
						+ torch.mul(diff_blue_y,torch.mul(diff_red_y,u3))+torch.mul(blue,torch.mul(diff_red_yy,u3))+torch.mul(blue,torch.mul(diff_red_y,diff_u3_y))) \
						+ torch.mul(omegaE_11[1,:,:],cov_dev_u[0,1,:,:])+torch.mul(omegaE_12[1,:,:],cov_dev_u[1,1,:,:])+torch.mul(omegaE_13[1,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_yy_1 = torch.unsqueeze(cov_dev_order2_yy_1,dim=0) #[1xHxW]
	
	cov_dev_order2_1 = torch.cat((cov_dev_order2_xx_1,cov_dev_order2_xy_1,cov_dev_order2_yx_1,cov_dev_order2_yy_1),dim=0) #[4xHxW]
	cov_dev_order2_1 = torch.unsqueeze(cov_dev_order2_1,dim=0) #[1x4xHxW]			

	cov_dev_order2_xx_2 = diff_u2_xx + torch.mul(-tmpx,torch.mul(omegaE_21[0,:,:],u1)+torch.mul(omegaE_22[0,:,:],u2)+torch.mul(omegaE_23[0,:,:],u3)) \
						- torch.mul((1./den),torch.mul(diff_red_x,torch.mul(diff_green_x,u1))+torch.mul(red,torch.mul(diff_green_xx,u1))+torch.mul(red,torch.mul(diff_green_x,diff_u1_x)) \
						+ torch.mul(diff_green_x,torch.mul(diff_green_x,u2))+torch.mul(green,torch.mul(diff_green_xx,u2))+torch.mul(green,torch.mul(diff_green_x,diff_u2_x))\
						+ torch.mul(diff_blue_x,torch.mul(diff_green_x,u3))+torch.mul(blue,torch.mul(diff_green_xx,u3))+torch.mul(blue,torch.mul(diff_green_x,diff_u3_x))) \
						+ torch.mul(omegaE_21[0,:,:],cov_dev_u[0,0,:,:])+torch.mul(omegaE_22[0,:,:],cov_dev_u[1,0,:,:])+torch.mul(omegaE_23[0,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_xx_2 = torch.unsqueeze(cov_dev_order2_xx_2,dim=0) #[1xHxW]
	
	cov_dev_order2_xy_2 = diff_u2_xy + torch.mul(-tmpx,torch.mul(omegaE_21[1,:,:],u1)+torch.mul(omegaE_22[1,:,:],u2)+torch.mul(omegaE_23[1,:,:],u3)) \
						- torch.mul((1./den),torch.mul(diff_red_x,torch.mul(diff_green_y,u1))+torch.mul(red,torch.mul(diff_green_xy,u1))+torch.mul(red,torch.mul(diff_green_y,diff_u1_x)) \
						+ torch.mul(diff_green_x,torch.mul(diff_green_y,u2))+torch.mul(green,torch.mul(diff_green_xy,u2))+torch.mul(green,torch.mul(diff_green_y,diff_u2_x))\
						+ torch.mul(diff_blue_x,torch.mul(diff_green_y,u3))+torch.mul(blue,torch.mul(diff_green_xy,u3))+torch.mul(blue,torch.mul(diff_green_y,diff_u3_x))) \
						+ torch.mul(omegaE_21[0,:,:],cov_dev_u[0,1,:,:])+torch.mul(omegaE_22[0,:,:],cov_dev_u[1,1,:,:])+torch.mul(omegaE_23[0,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_xy_2 = torch.unsqueeze(cov_dev_order2_xy_2,dim=0) #[1xHxW]

	cov_dev_order2_yx_2 = diff_u2_xy + torch.mul(-tmpy,torch.mul(omegaE_21[0,:,:],u1)+torch.mul(omegaE_22[0,:,:],u2)+torch.mul(omegaE_23[0,:,:],u3)) \
						- torch.mul((1./den),torch.mul(diff_red_y,torch.mul(diff_green_x,u1))+torch.mul(red,torch.mul(diff_green_xy,u1))+torch.mul(red,torch.mul(diff_green_x,diff_u1_y)) \
						+ torch.mul(diff_green_y,torch.mul(diff_green_x,u2))+torch.mul(green,torch.mul(diff_green_xy,u2))+torch.mul(green,torch.mul(diff_green_x,diff_u2_y))\
						+ torch.mul(diff_blue_y,torch.mul(diff_green_x,u3))+torch.mul(blue,torch.mul(diff_green_xy,u3))+torch.mul(blue,torch.mul(diff_green_x,diff_u3_y))) \
						+ torch.mul(omegaE_21[1,:,:],cov_dev_u[0,0,:,:])+torch.mul(omegaE_22[1,:,:],cov_dev_u[1,0,:,:])+torch.mul(omegaE_23[1,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_yx_2 = torch.unsqueeze(cov_dev_order2_yx_2,dim=0) #[1xHxW]

	cov_dev_order2_yy_2 = diff_u2_yy + torch.mul(-tmpy,torch.mul(omegaE_21[1,:,:],u1)+torch.mul(omegaE_22[1,:,:],u2)+torch.mul(omegaE_23[1,:,:],u3)) \
						- torch.mul((1./den),torch.mul(diff_red_y,torch.mul(diff_green_y,u1))+torch.mul(red,torch.mul(diff_green_yy,u1))+torch.mul(red,torch.mul(diff_green_y,diff_u1_y)) \
						+ torch.mul(diff_green_y,torch.mul(diff_green_y,u2))+torch.mul(green,torch.mul(diff_green_yy,u2))+torch.mul(green,torch.mul(diff_green_y,diff_u2_y))\
						+ torch.mul(diff_blue_y,torch.mul(diff_green_y,u3))+torch.mul(blue,torch.mul(diff_green_yy,u3))+torch.mul(blue,torch.mul(diff_green_y,diff_u3_y))) \
						+ torch.mul(omegaE_21[1,:,:],cov_dev_u[0,1,:,:])+torch.mul(omegaE_22[1,:,:],cov_dev_u[1,1,:,:])+torch.mul(omegaE_23[1,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_yy_2 = torch.unsqueeze(cov_dev_order2_yy_2,dim=0) #[1xHxW]	

	cov_dev_order2_2 = torch.cat((cov_dev_order2_xx_2,cov_dev_order2_xy_2,cov_dev_order2_yx_2,cov_dev_order2_yy_2),dim=0) #[4xHxW]
	cov_dev_order2_2 = torch.unsqueeze(cov_dev_order2_2,dim=0) #[1x4xHxW]		


	cov_dev_order2_xx_3 = diff_u3_xx + torch.mul(-tmpx,torch.mul(omegaE_31[0,:,:],u1)+torch.mul(omegaE_32[0,:,:],u2)+torch.mul(omegaE_33[0,:,:],u3)) \
						- torch.mul((1./den),torch.mul(diff_red_x,torch.mul(diff_blue_x,u1))+torch.mul(red,torch.mul(diff_blue_xx,u1))+torch.mul(red,torch.mul(diff_blue_x,diff_u1_x)) \
						+ torch.mul(diff_green_x,torch.mul(diff_blue_x,u2))+torch.mul(green,torch.mul(diff_blue_xx,u2))+torch.mul(green,torch.mul(diff_blue_x,diff_u2_x))\
						+ torch.mul(diff_blue_x,torch.mul(diff_blue_x,u3))+torch.mul(blue,torch.mul(diff_blue_xx,u3))+torch.mul(blue,torch.mul(diff_blue_x,diff_u3_x))) \
						+ torch.mul(omegaE_31[0,:,:],cov_dev_u[0,0,:,:])+torch.mul(omegaE_32[0,:,:],cov_dev_u[1,0,:,:])+torch.mul(omegaE_33[0,:,:],cov_dev_u[2,0,:,:])					
	cov_dev_order2_xx_3 = torch.unsqueeze(cov_dev_order2_xx_3,dim=0) #[1xHxW]					

	cov_dev_order2_xy_3 = diff_u3_xy + torch.mul(-tmpx,torch.mul(omegaE_31[1,:,:],u1)+torch.mul(omegaE_32[1,:,:],u2)+torch.mul(omegaE_33[1,:,:],u3)) \
						- torch.mul((1./den),torch.mul(diff_red_x,torch.mul(diff_blue_y,u1))+torch.mul(red,torch.mul(diff_blue_xy,u1))+torch.mul(red,torch.mul(diff_blue_y,diff_u1_x)) \
						+ torch.mul(diff_green_x,torch.mul(diff_blue_y,u2))+torch.mul(green,torch.mul(diff_blue_xy,u2))+torch.mul(green,torch.mul(diff_blue_y,diff_u2_x))\
						+ torch.mul(diff_blue_x,torch.mul(diff_blue_y,u3))+torch.mul(blue,torch.mul(diff_blue_xy,u3))+torch.mul(blue,torch.mul(diff_blue_y,diff_u3_x))) \
						+ torch.mul(omegaE_31[0,:,:],cov_dev_u[0,1,:,:])+torch.mul(omegaE_32[0,:,:],cov_dev_u[1,1,:,:])+torch.mul(omegaE_33[0,:,:],cov_dev_u[2,1,:,:])					
	cov_dev_order2_xy_3 = torch.unsqueeze(cov_dev_order2_xy_3,dim=0) #[1xHxW]
						
	cov_dev_order2_yx_3 = diff_u3_xy + torch.mul(-tmpy,torch.mul(omegaE_31[0,:,:],u1)+torch.mul(omegaE_32[0,:,:],u2)+torch.mul(omegaE_33[0,:,:],u3)) \
						- torch.mul((1./den),torch.mul(diff_red_y,torch.mul(diff_blue_x,u1))+torch.mul(red,torch.mul(diff_blue_xy,u1))+torch.mul(red,torch.mul(diff_blue_x,diff_u1_y)) \
						+ torch.mul(diff_green_y,torch.mul(diff_blue_x,u2))+torch.mul(green,torch.mul(diff_blue_xy,u2))+torch.mul(green,torch.mul(diff_blue_x,diff_u2_y))\
						+ torch.mul(diff_blue_y,torch.mul(diff_blue_x,u3))+torch.mul(blue,torch.mul(diff_blue_xy,u3))+torch.mul(blue,torch.mul(diff_blue_x,diff_u3_y))) \
						+ torch.mul(omegaE_31[1,:,:],cov_dev_u[0,0,:,:])+torch.mul(omegaE_32[1,:,:],cov_dev_u[1,0,:,:])+torch.mul(omegaE_33[1,:,:],cov_dev_u[2,0,:,:])					
	cov_dev_order2_yx_3 = torch.unsqueeze(cov_dev_order2_yx_3,dim=0) #[1xHxW]

	cov_dev_order2_yy_3 = diff_u3_yy + torch.mul(-tmpy,torch.mul(omegaE_31[1,:,:],u1)+torch.mul(omegaE_32[1,:,:],u2)+torch.mul(omegaE_33[1,:,:],u3)) \
						- torch.mul((1./den),torch.mul(diff_red_y,torch.mul(diff_blue_y,u1))+torch.mul(red,torch.mul(diff_blue_yy,u1))+torch.mul(red,torch.mul(diff_blue_y,diff_u1_y)) \
						+ torch.mul(diff_green_y,torch.mul(diff_blue_y,u2))+torch.mul(green,torch.mul(diff_blue_yy,u2))+torch.mul(green,torch.mul(diff_blue_y,diff_u2_y))\
						+ torch.mul(diff_blue_y,torch.mul(diff_blue_y,u3))+torch.mul(blue,torch.mul(diff_blue_yy,u3))+torch.mul(blue,torch.mul(diff_blue_y,diff_u3_y))) \
						+ torch.mul(omegaE_31[1,:,:],cov_dev_u[0,1,:,:])+torch.mul(omegaE_32[1,:,:],cov_dev_u[1,1,:,:])+torch.mul(omegaE_33[1,:,:],cov_dev_u[2,1,:,:])					
	cov_dev_order2_yy_3 = torch.unsqueeze(cov_dev_order2_yy_3,dim=0) #[1xHxW]	

	cov_dev_order2_3 = torch.cat((cov_dev_order2_xx_3,cov_dev_order2_xy_3,cov_dev_order2_yx_3,cov_dev_order2_yy_3),dim=0) #[4xHxW]
	cov_dev_order2_3 = torch.unsqueeze(cov_dev_order2_3,dim=0) #[1x4xHxW]					


	cov_dev_order2 = torch.cat((cov_dev_order2_1,cov_dev_order2_2,cov_dev_order2_3),dim=0) #[3x4xHxW]

	return cov_dev_order2 #[3x4xHxW]

def covariant_derivative_first_order_GL3_optimal(u, omegaE_11,omegaE_12,omegaE_13,omegaE_21,omegaE_22,omegaE_23,omegaE_31,omegaE_32,omegaE_33,dtype):

	u1 = torch.narrow(u,1,0,1)       # [1x1xHxW]
	u1 = torch.squeeze(u1)   		 # [HxW]
	u1 = torch.unsqueeze(u1,dim=0)   # [1xHxW]
	u1tilde = u1.repeat(2,1,1)   	 # [2xHxW]
	u2 = torch.narrow(u,1,1,1)       # [1x1xHxW]
	u2 = torch.squeeze(u2)   		 # [HxW]
	u2 = torch.unsqueeze(u2,dim=0)   # [1xHxW]
	u2tilde = u2.repeat(2,1,1)   	 # [2xHxW]
	u3 = torch.narrow(u,1,2,1)       # [1x1xHxW]
	u3 = torch.squeeze(u3)   		 # [HxW]
	u3 = torch.unsqueeze(u3,dim=0)   # [1xHxW]
	u3tilde = u3.repeat(2,1,1)       # [2xHxW]


	diff_u1_x = derivative_central_x_greylevel(torch.squeeze(u1),dtype)     #[HxW]
	diff_u1_x = torch.unsqueeze(diff_u1_x,dim=0)							#[1xHxW]
	diff_u1_y = derivative_central_y_greylevel(torch.squeeze(u1),dtype)     #[HxW]
	diff_u1_y = torch.unsqueeze(diff_u1_y,dim=0)							#[1xHxW]
	diff_u1 = torch.cat((diff_u1_x,diff_u1_y),dim=0)                        #[2xHxW]
	diff_u1tilde = torch.unsqueeze(diff_u1,dim=0)								#[1x2xHxW]

	diff_u2_x = derivative_central_x_greylevel(torch.squeeze(u2),dtype)     #[HxW]
	diff_u2_x = torch.unsqueeze(diff_u2_x,dim=0)							#[1xHxW]
	diff_u2_y = derivative_central_y_greylevel(torch.squeeze(u2),dtype)     #[HxW]
	diff_u2_y = torch.unsqueeze(diff_u2_y,dim=0)							#[1xHxW]
	diff_u2 = torch.cat((diff_u2_x,diff_u2_y),dim=0)                        #[2xHxW]
	diff_u2tilde = torch.unsqueeze(diff_u2,dim=0)								#[1x2xHxW]

	diff_u3_x = derivative_central_x_greylevel(torch.squeeze(u3),dtype)     #[HxW]
	diff_u3_x = torch.unsqueeze(diff_u3_x,dim=0)							#[1xHxW]
	diff_u3_y = derivative_central_y_greylevel(torch.squeeze(u3),dtype)     #[HxW]
	diff_u3_y = torch.unsqueeze(diff_u3_y,dim=0)							#[1xHxW]
	diff_u3 = torch.cat((diff_u3_x,diff_u3_y),dim=0)                        #[2xHxW]
	diff_u3tilde = torch.unsqueeze(diff_u3,dim=0)								#[1x2xHxW]

	diff_u = torch.cat((diff_u1tilde,diff_u2tilde,diff_u3tilde),dim=0)			#[3x2xHxW]

	
	
	coeffrep_num = - torch.sum( torch.mul(diff_u1+0.5*(torch.mul(omegaE_13-omegaE_31,u3tilde))+0.5*(torch.mul(omegaE_12-omegaE_21,u2tilde)),2.*torch.mul(omegaE_11,u1tilde)+torch.mul(omegaE_12+omegaE_21,u2tilde)+torch.mul(omegaE_13+omegaE_31,u3tilde)),dim=0) \
					- torch.sum( torch.mul(diff_u2+0.5*(torch.mul(omegaE_21-omegaE_12,u1tilde))+0.5*(torch.mul(omegaE_23-omegaE_32,u3tilde)),2.*torch.mul(omegaE_22,u2tilde)+torch.mul(omegaE_12+omegaE_21,u1tilde)+torch.mul(omegaE_23+omegaE_32,u3tilde)),dim=0) \
					- torch.sum( torch.mul(diff_u3+0.5*(torch.mul(omegaE_31-omegaE_13,u1tilde))+0.5*(torch.mul(omegaE_32-omegaE_23,u2tilde)),2.*torch.mul(omegaE_33,u3tilde)+torch.mul(omegaE_23+omegaE_32,u2tilde)+torch.mul(omegaE_13+omegaE_31,u1tilde)),dim=0) #[HxW]	
					

	coeffrep_den = 2.*torch.sum(torch.mul(torch.mul(omegaE_11,u1tilde),torch.mul(omegaE_11,u1tilde)+torch.mul(omegaE_12+omegaE_21,u2tilde)+torch.mul(omegaE_13+omegaE_31,u3tilde)),dim=0) \
					 + 2.*torch.sum(torch.mul(torch.mul(omegaE_22,u2tilde),torch.mul(omegaE_22,u2tilde)+torch.mul(omegaE_23+omegaE_32,u3tilde)+torch.mul(omegaE_12+omegaE_21,u1tilde)),dim=0)\
					 + 2.*torch.sum(torch.mul(torch.mul(omegaE_33,u3tilde),torch.mul(omegaE_33,u3tilde)+torch.mul(omegaE_13+omegaE_31,u1tilde)+torch.mul(omegaE_23+omegaE_32,u2tilde)),dim=0) \
					+ 0.5*torch.sum(torch.square( torch.mul(omegaE_12+omegaE_21,u1tilde)+torch.mul(omegaE_23+omegaE_32,u3tilde)),dim=0) \
					+ 0.5*torch.sum(torch.square( torch.mul(omegaE_12+omegaE_21,u2tilde)+torch.mul(omegaE_13+omegaE_31,u3tilde)),dim=0) \
					+ 0.5*torch.sum(torch.square( torch.mul(omegaE_13+omegaE_31,u1tilde)+torch.mul(omegaE_23+omegaE_32,u2tilde)),dim=0) #[HxW]


	coeff_rep = torch.div(coeffrep_num,0.001+coeffrep_den) #[HxW]
	coeff_rep_tmp = torch.unsqueeze(coeff_rep,dim=0)
	coeff_rep_tilde = coeff_rep_tmp.repeat(2,1,1) #[2xHxW]
	
	gammaconnection11 = torch.mul(coeff_rep,omegaE_11) # [2xHxW]
	gammaconnection12 = 0.5*(omegaE_12-omegaE_21)+0.5*torch.mul(coeff_rep_tilde,omegaE_12+omegaE_21) # [2xHxW]
	gammaconnection13 = 0.5*(omegaE_13-omegaE_31)+0.5*torch.mul(coeff_rep_tilde,omegaE_13+omegaE_31) # [2xHxW]
	gammaconnection21 = -0.5*(omegaE_12-omegaE_21)+0.5*torch.mul(coeff_rep_tilde,omegaE_12+omegaE_21) # [2xHxW]
	gammaconnection22 = torch.mul(coeff_rep,omegaE_22) # [2xHxW]
	gammaconnection23 = 0.5*(omegaE_23-omegaE_32)+0.5*torch.mul(coeff_rep_tilde,omegaE_23+omegaE_32) # [2xHxW]
	gammaconnection31 = -0.5*(omegaE_13-omegaE_31)+0.5*torch.mul(coeff_rep_tilde,omegaE_13+omegaE_31) # [2xHxW]
	gammaconnection32 = -0.5*(omegaE_32-omegaE_23)+0.5*torch.mul(coeff_rep_tilde,omegaE_23+omegaE_32) # [2xHxW]
	gammaconnection33 = torch.mul(coeff_rep,omegaE_33) # [2xHxW]
		

	zero_order_term1 = torch.mul(gammaconnection11,u1tilde) + torch.mul(gammaconnection12,u2tilde) + torch.mul(gammaconnection13,u3tilde) # [2xHxW]
	zero_order_term1 = torch.unsqueeze(zero_order_term1,dim=0) #[1x2xHxW]
	zero_order_term2 = torch.mul(gammaconnection21,u1tilde) + torch.mul(gammaconnection22,u2tilde) + torch.mul(gammaconnection23,u3tilde) # [2xHxW]
	zero_order_term2 = torch.unsqueeze(zero_order_term2,dim=0) #[1x2xHxW]
	zero_order_term3 = torch.mul(gammaconnection31,u1tilde) + torch.mul(gammaconnection32,u2tilde) + torch.mul(gammaconnection33,u3tilde) # [2xHxW]
	zero_order_term3 = torch.unsqueeze(zero_order_term3,dim=0) #[1x2xHxW]
	zero_order_term  = torch.cat((zero_order_term1,zero_order_term2,zero_order_term3),dim=0) #[3x2xHxW]


	cov_dev_u = diff_u + zero_order_term  #[3x2xHxW]	

	return cov_dev_u 	


def connection1form_GL3_optimal(u, omegaE_11,omegaE_12,omegaE_13,omegaE_21,omegaE_22,omegaE_23,omegaE_31,omegaE_32,omegaE_33,dtype):

	u1 = torch.narrow(u,1,0,1)       # [1x1xHxW]
	u1 = torch.squeeze(u1)   		 # [HxW]
	u1 = torch.unsqueeze(u1,dim=0)   # [1xHxW]
	u1tilde = u1.repeat(2,1,1)   	 # [2xHxW]
	u2 = torch.narrow(u,1,1,1)       # [1x1xHxW]
	u2 = torch.squeeze(u2)   		 # [HxW]
	u2 = torch.unsqueeze(u2,dim=0)   # [1xHxW]
	u2tilde = u2.repeat(2,1,1)   	 # [2xHxW]
	u3 = torch.narrow(u,1,2,1)       # [1x1xHxW]
	u3 = torch.squeeze(u3)   		 # [HxW]
	u3 = torch.unsqueeze(u3,dim=0)   # [1xHxW]
	u3tilde = u3.repeat(2,1,1)       # [2xHxW]


	diff_u1_x = derivative_central_x_greylevel(torch.squeeze(u1),dtype)     #[HxW]
	diff_u1_x = torch.unsqueeze(diff_u1_x,dim=0)							#[1xHxW]
	diff_u1_y = derivative_central_y_greylevel(torch.squeeze(u1),dtype)     #[HxW]
	diff_u1_y = torch.unsqueeze(diff_u1_y,dim=0)							#[1xHxW]
	diff_u1 = torch.cat((diff_u1_x,diff_u1_y),dim=0)                        #[2xHxW]
	diff_u1tilde = torch.unsqueeze(diff_u1,dim=0)								#[1x2xHxW]

	diff_u2_x = derivative_central_x_greylevel(torch.squeeze(u2),dtype)     #[HxW]
	diff_u2_x = torch.unsqueeze(diff_u2_x,dim=0)							#[1xHxW]
	diff_u2_y = derivative_central_y_greylevel(torch.squeeze(u2),dtype)     #[HxW]
	diff_u2_y = torch.unsqueeze(diff_u2_y,dim=0)							#[1xHxW]
	diff_u2 = torch.cat((diff_u2_x,diff_u2_y),dim=0)                        #[2xHxW]
	diff_u2tilde = torch.unsqueeze(diff_u2,dim=0)								#[1x2xHxW]

	diff_u3_x = derivative_central_x_greylevel(torch.squeeze(u3),dtype)     #[HxW]
	diff_u3_x = torch.unsqueeze(diff_u3_x,dim=0)							#[1xHxW]
	diff_u3_y = derivative_central_y_greylevel(torch.squeeze(u3),dtype)     #[HxW]
	diff_u3_y = torch.unsqueeze(diff_u3_y,dim=0)							#[1xHxW]
	diff_u3 = torch.cat((diff_u3_x,diff_u3_y),dim=0)                        #[2xHxW]
	diff_u3tilde = torch.unsqueeze(diff_u3,dim=0)								#[1x2xHxW]

	diff_u = torch.cat((diff_u1tilde,diff_u2tilde,diff_u3tilde),dim=0)			#[3x2xHxW]

	
	
	coeffrep_num = - torch.sum( torch.mul(diff_u1+0.5*(torch.mul(omegaE_13-omegaE_31,u3tilde))+0.5*(torch.mul(omegaE_12-omegaE_21,u2tilde)),2.*torch.mul(omegaE_11,u1tilde)+torch.mul(omegaE_12+omegaE_21,u2tilde)+torch.mul(omegaE_13+omegaE_31,u3tilde)),dim=0) \
					- torch.sum( torch.mul(diff_u2+0.5*(torch.mul(omegaE_21-omegaE_12,u1tilde))+0.5*(torch.mul(omegaE_23-omegaE_32,u3tilde)),2.*torch.mul(omegaE_22,u2tilde)+torch.mul(omegaE_12+omegaE_21,u1tilde)+torch.mul(omegaE_23+omegaE_32,u3tilde)),dim=0) \
					- torch.sum( torch.mul(diff_u3+0.5*(torch.mul(omegaE_31-omegaE_13,u1tilde))+0.5*(torch.mul(omegaE_32-omegaE_23,u2tilde)),2.*torch.mul(omegaE_33,u3tilde)+torch.mul(omegaE_23+omegaE_32,u2tilde)+torch.mul(omegaE_13+omegaE_31,u1tilde)),dim=0) #[HxW]	
					

	coeffrep_den = 2.*torch.sum(torch.mul(torch.mul(omegaE_11,u1tilde),torch.mul(omegaE_11,u1tilde)+torch.mul(omegaE_12+omegaE_21,u2tilde)+torch.mul(omegaE_13+omegaE_31,u3tilde)),dim=0) \
					 + 2.*torch.sum(torch.mul(torch.mul(omegaE_22,u2tilde),torch.mul(omegaE_22,u2tilde)+torch.mul(omegaE_23+omegaE_32,u3tilde)+torch.mul(omegaE_12+omegaE_21,u1tilde)),dim=0)\
					 + 2.*torch.sum(torch.mul(torch.mul(omegaE_33,u3tilde),torch.mul(omegaE_33,u3tilde)+torch.mul(omegaE_13+omegaE_31,u1tilde)+torch.mul(omegaE_23+omegaE_32,u2tilde)),dim=0) \
					+ 0.5*torch.sum(torch.square( torch.mul(omegaE_12+omegaE_21,u1tilde)+torch.mul(omegaE_23+omegaE_32,u3tilde)),dim=0) \
					+ 0.5*torch.sum(torch.square( torch.mul(omegaE_12+omegaE_21,u2tilde)+torch.mul(omegaE_13+omegaE_31,u3tilde)),dim=0) \
					+ 0.5*torch.sum(torch.square( torch.mul(omegaE_13+omegaE_31,u1tilde)+torch.mul(omegaE_23+omegaE_32,u2tilde)),dim=0) #[HxW]


	coeff_rep = torch.div(coeffrep_num,0.001+coeffrep_den) #[HxW]
	coeff_rep_tmp = torch.unsqueeze(coeff_rep,dim=0)
	coeff_rep_tilde = coeff_rep_tmp.repeat(2,1,1) #[2xHxW]
	
	gammaconnection11 = torch.mul(coeff_rep,omegaE_11) # [2xHxW]
	gammaconnection12 = 0.5*(omegaE_12-omegaE_21)+0.5*torch.mul(coeff_rep_tilde,omegaE_12+omegaE_21) # [2xHxW]
	gammaconnection13 = 0.5*(omegaE_13-omegaE_31)+0.5*torch.mul(coeff_rep_tilde,omegaE_13+omegaE_31) # [2xHxW]
	gammaconnection21 = -0.5*(omegaE_12-omegaE_21)+0.5*torch.mul(coeff_rep_tilde,omegaE_12+omegaE_21) # [2xHxW]
	gammaconnection22 = torch.mul(coeff_rep,omegaE_22) # [2xHxW]
	gammaconnection23 = 0.5*(omegaE_23-omegaE_32)+0.5*torch.mul(coeff_rep_tilde,omegaE_23+omegaE_32) # [2xHxW]
	gammaconnection31 = -0.5*(omegaE_13-omegaE_31)+0.5*torch.mul(coeff_rep_tilde,omegaE_13+omegaE_31) # [2xHxW]
	gammaconnection32 = -0.5*(omegaE_32-omegaE_23)+0.5*torch.mul(coeff_rep_tilde,omegaE_23+omegaE_32) # [2xHxW]
	gammaconnection33 = torch.mul(coeff_rep,omegaE_33) # [2xHxW]

	return gammaconnection11,gammaconnection12,gammaconnection13,gammaconnection21,gammaconnection22,gammaconnection23,gammaconnection31,gammaconnection32,gammaconnection33


def covariant_derivative_second_order_GL3_optimal(u, omegaE_11,omegaE_12,omegaE_13,omegaE_21,omegaE_22,omegaE_23,omegaE_31,omegaE_32,omegaE_33,omegaTM_11,omegaTM_12,omegaTM_21,omegaTM_22,dtype):
	
	cov_dev_u = covariant_derivative_first_order_GL3_optimal(u, omegaE_11,omegaE_12,omegaE_13,omegaE_21,omegaE_22,omegaE_23,omegaE_31,omegaE_32,omegaE_33,dtype)
	gammaconnection_11,gammaconnection_12,gammaconnection_13,gammaconnection_21,gammaconnection_22,gammaconnection_23,gammaconnection_31,gammaconnection_32,gammaconnection_33 = connection1form_GL3_optimal(u, omegaE_11,omegaE_12,omegaE_13,omegaE_21,omegaE_22,omegaE_23,omegaE_31,omegaE_32,omegaE_33,dtype)

	u1 = torch.narrow(u,1,0,1)       # [1x1xHxW]
	u1 = torch.squeeze(u1)   		 # [HxW]
	u2 = torch.narrow(u,1,1,1)       # [1x1xHxW]
	u2 = torch.squeeze(u2)   		 # [HxW]
	u3 = torch.narrow(u,1,2,1)       # [1x1xHxW]
	u3 = torch.squeeze(u3)   		 # [HxW]

	diff_u1_x = derivative_central_x_greylevel(u1,dtype) #[HxW]
	diff_u1_y = derivative_central_x_greylevel(u1,dtype) #[HxW]


	diff_u2_x = derivative_central_x_greylevel(u2,dtype) #[HxW]
	diff_u2_y = derivative_central_x_greylevel(u2,dtype) #[HxW]

	diff_u3_x = derivative_central_x_greylevel(u3,dtype) #[HxW]
	diff_u3_y = derivative_central_x_greylevel(u3,dtype) #[HxW]

	diff_u1_xx = derivative_xx_greylevel(u1,dtype) #[HxW]
	diff_u1_xy = derivative_xy_greylevel(u1,dtype) #[HxW]
	diff_u1_yy = derivative_yy_greylevel(u1,dtype) #[HxW]

	diff_u2_xx = derivative_xx_greylevel(u2,dtype) #[HxW]
	diff_u2_xy = derivative_xy_greylevel(u2,dtype) #[HxW]
	diff_u2_yy = derivative_yy_greylevel(u2,dtype) #[HxW]

	diff_u3_xx = derivative_xx_greylevel(u3,dtype) #[HxW]
	diff_u3_xy = derivative_xy_greylevel(u3,dtype) #[HxW]
	diff_u3_yy = derivative_yy_greylevel(u3,dtype) #[HxW]

	cov_dev_order2_xx_1 = diff_u1_xx + torch.mul(derivative_central_x_greylevel(gammaconnection_11[0,:,:],dtype),u1)+torch.mul(gammaconnection_11[0,:,:],diff_u1_x) \
						+ torch.mul(derivative_central_x_greylevel(gammaconnection_12[0,:,:],dtype),u2)+torch.mul(gammaconnection_12[0,:,:],diff_u2_x) \
						+ torch.mul(derivative_central_x_greylevel(gammaconnection_13[0,:,:],dtype),u3)+torch.mul(gammaconnection_13[0,:,:],diff_u3_x) \
						+ torch.mul(gammaconnection_11[0,:,:],cov_dev_u[0,0,:,:])+torch.mul(gammaconnection_12[0,:,:],cov_dev_u[1,0,:,:])+torch.mul(gammaconnection_13[0,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_xx_1 = torch.unsqueeze(cov_dev_order2_xx_1,dim=0) #[1xHxW]

	cov_dev_order2_xy_1 = diff_u1_xy + torch.mul(derivative_central_x_greylevel(gammaconnection_11[1,:,:],dtype),u1)+torch.mul(gammaconnection_11[1,:,:],diff_u1_x) \
						+ torch.mul(derivative_central_x_greylevel(gammaconnection_12[1,:,:],dtype),u2)+torch.mul(gammaconnection_12[1,:,:],diff_u2_x) \
						+ torch.mul(derivative_central_x_greylevel(gammaconnection_13[1,:,:],dtype),u3)+torch.mul(gammaconnection_13[1,:,:],diff_u3_x) \
						+ torch.mul(gammaconnection_11[0,:,:],cov_dev_u[0,1,:,:])+torch.mul(gammaconnection_12[0,:,:],cov_dev_u[1,1,:,:])+torch.mul(gammaconnection_13[0,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_xy_1 = torch.unsqueeze(cov_dev_order2_xy_1,dim=0) #[1xHxW]

	cov_dev_order2_yx_1 = diff_u1_xy + torch.mul(derivative_central_y_greylevel(gammaconnection_11[0,:,:],dtype),u1)+torch.mul(gammaconnection_11[0,:,:],diff_u1_y) \
						+ torch.mul(derivative_central_y_greylevel(gammaconnection_12[0,:,:],dtype),u2)+torch.mul(gammaconnection_12[0,:,:],diff_u2_y) \
						+ torch.mul(derivative_central_y_greylevel(gammaconnection_13[0,:,:],dtype),u3)+torch.mul(gammaconnection_13[0,:,:],diff_u3_y) \
						+ torch.mul(gammaconnection_11[1,:,:],cov_dev_u[0,0,:,:])+torch.mul(gammaconnection_12[1,:,:],cov_dev_u[1,0,:,:])+torch.mul(gammaconnection_13[1,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_yx_1 = torch.unsqueeze(cov_dev_order2_yx_1,dim=0) #[1xHxW]

	cov_dev_order2_yy_1 = diff_u1_yy + torch.mul(derivative_central_y_greylevel(gammaconnection_11[1,:,:],dtype),u1)+torch.mul(gammaconnection_11[1,:,:],diff_u1_y) \
						+ torch.mul(derivative_central_y_greylevel(gammaconnection_12[1,:,:],dtype),u2)+torch.mul(gammaconnection_12[1,:,:],diff_u2_y) \
						+ torch.mul(derivative_central_y_greylevel(gammaconnection_13[1,:,:],dtype),u3)+torch.mul(gammaconnection_13[1,:,:],diff_u3_y) \
						+ torch.mul(gammaconnection_11[1,:,:],cov_dev_u[0,1,:,:])+torch.mul(gammaconnection_12[1,:,:],cov_dev_u[1,1,:,:])+torch.mul(gammaconnection_13[1,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_yy_1 = torch.unsqueeze(cov_dev_order2_yy_1,dim=0) #[1xHxW]

	cov_dev_order2_1 = torch.cat((cov_dev_order2_xx_1,cov_dev_order2_xy_1,cov_dev_order2_yx_1,cov_dev_order2_yy_1),dim=0) #[4xHxW]
	cov_dev_order2_1 = torch.unsqueeze(cov_dev_order2_1,dim=0) #[1x4xHxW]


	cov_dev_order2_xx_2 = diff_u2_xx + torch.mul(derivative_central_x_greylevel(gammaconnection_21[0,:,:],dtype),u1)+torch.mul(gammaconnection_21[0,:,:],diff_u1_x) \
						+ torch.mul(derivative_central_x_greylevel(gammaconnection_22[0,:,:],dtype),u2)+torch.mul(gammaconnection_22[0,:,:],diff_u2_x) \
						+ torch.mul(derivative_central_x_greylevel(gammaconnection_23[0,:,:],dtype),u3)+torch.mul(gammaconnection_23[0,:,:],diff_u3_x) \
						+ torch.mul(gammaconnection_21[0,:,:],cov_dev_u[0,0,:,:])+torch.mul(gammaconnection_22[0,:,:],cov_dev_u[1,0,:,:])+torch.mul(gammaconnection_23[0,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_xx_2 = torch.unsqueeze(cov_dev_order2_xx_2,dim=0) #[1xHxW]

	cov_dev_order2_xy_2 = diff_u2_xy + torch.mul(derivative_central_x_greylevel(gammaconnection_21[1,:,:],dtype),u1)+torch.mul(gammaconnection_21[1,:,:],diff_u1_x) \
						+ torch.mul(derivative_central_x_greylevel(gammaconnection_22[1,:,:],dtype),u2)+torch.mul(gammaconnection_22[1,:,:],diff_u2_x) \
						+ torch.mul(derivative_central_x_greylevel(gammaconnection_23[1,:,:],dtype),u3)+torch.mul(gammaconnection_23[1,:,:],diff_u3_x) \
						+ torch.mul(gammaconnection_21[0,:,:],cov_dev_u[0,1,:,:])+torch.mul(gammaconnection_22[0,:,:],cov_dev_u[1,1,:,:])+torch.mul(gammaconnection_23[0,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_xy_2 = torch.unsqueeze(cov_dev_order2_xy_2,dim=0) #[1xHxW]

	cov_dev_order2_yx_2 = diff_u2_xy + torch.mul(derivative_central_y_greylevel(gammaconnection_21[0,:,:],dtype),u1)+torch.mul(gammaconnection_21[0,:,:],diff_u1_y) \
						+ torch.mul(derivative_central_y_greylevel(gammaconnection_22[0,:,:],dtype),u2)+torch.mul(gammaconnection_22[0,:,:],diff_u2_y) \
						+ torch.mul(derivative_central_y_greylevel(gammaconnection_23[0,:,:],dtype),u3)+torch.mul(gammaconnection_23[0,:,:],diff_u3_y) \
						+ torch.mul(gammaconnection_21[1,:,:],cov_dev_u[0,0,:,:])+torch.mul(gammaconnection_22[1,:,:],cov_dev_u[1,0,:,:])+torch.mul(gammaconnection_23[1,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_yx_2 = torch.unsqueeze(cov_dev_order2_yx_2,dim=0) #[1xHxW]

	cov_dev_order2_yy_2 = diff_u2_yy + torch.mul(derivative_central_y_greylevel(gammaconnection_11[1,:,:],dtype),u1)+torch.mul(gammaconnection_11[1,:,:],diff_u1_y) \
						+ torch.mul(derivative_central_y_greylevel(gammaconnection_12[1,:,:],dtype),u2)+torch.mul(gammaconnection_12[1,:,:],diff_u2_y) \
						+ torch.mul(derivative_central_y_greylevel(gammaconnection_13[1,:,:],dtype),u3)+torch.mul(gammaconnection_13[1,:,:],diff_u3_y) \
						+ torch.mul(gammaconnection_21[1,:,:],cov_dev_u[0,1,:,:])+torch.mul(gammaconnection_22[1,:,:],cov_dev_u[1,1,:,:])+torch.mul(gammaconnection_23[1,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_yy_2 = torch.unsqueeze(cov_dev_order2_yy_2,dim=0) #[1xHxW]

	cov_dev_order2_2 = torch.cat((cov_dev_order2_xx_2,cov_dev_order2_xy_2,cov_dev_order2_yx_2,cov_dev_order2_yy_2),dim=0) #[4xHxW]
	cov_dev_order2_2 = torch.unsqueeze(cov_dev_order2_2,dim=0) #[1x4xHxW]

	cov_dev_order2_xx_3 = diff_u3_xx + torch.mul(derivative_central_x_greylevel(gammaconnection_31[0,:,:],dtype),u1)+torch.mul(gammaconnection_31[0,:,:],diff_u1_x) \
						+ torch.mul(derivative_central_x_greylevel(gammaconnection_32[0,:,:],dtype),u2)+torch.mul(gammaconnection_32[0,:,:],diff_u2_x) \
						+ torch.mul(derivative_central_x_greylevel(gammaconnection_33[0,:,:],dtype),u3)+torch.mul(gammaconnection_33[0,:,:],diff_u3_x) \
						+ torch.mul(gammaconnection_31[0,:,:],cov_dev_u[0,0,:,:])+torch.mul(gammaconnection_32[0,:,:],cov_dev_u[1,0,:,:])+torch.mul(gammaconnection_33[0,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_xx_3 = torch.unsqueeze(cov_dev_order2_xx_3,dim=0) #[1xHxW]

	cov_dev_order2_xy_3 = diff_u3_xy + torch.mul(derivative_central_x_greylevel(gammaconnection_31[1,:,:],dtype),u1)+torch.mul(gammaconnection_31[1,:,:],diff_u1_x) \
						+ torch.mul(derivative_central_x_greylevel(gammaconnection_32[1,:,:],dtype),u2)+torch.mul(gammaconnection_32[1,:,:],diff_u2_x) \
						+ torch.mul(derivative_central_x_greylevel(gammaconnection_33[1,:,:],dtype),u3)+torch.mul(gammaconnection_33[1,:,:],diff_u3_x) \
						+ torch.mul(gammaconnection_31[0,:,:],cov_dev_u[0,1,:,:])+torch.mul(gammaconnection_32[0,:,:],cov_dev_u[1,1,:,:])+torch.mul(gammaconnection_33[0,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_xy_3 = torch.unsqueeze(cov_dev_order2_xy_3,dim=0) #[1xHxW]

	cov_dev_order2_yx_3 = diff_u3_xy + torch.mul(derivative_central_y_greylevel(gammaconnection_31[0,:,:],dtype),u1)+torch.mul(gammaconnection_31[0,:,:],diff_u1_y) \
						+ torch.mul(derivative_central_y_greylevel(gammaconnection_32[0,:,:],dtype),u2)+torch.mul(gammaconnection_32[0,:,:],diff_u2_y) \
						+ torch.mul(derivative_central_y_greylevel(gammaconnection_33[0,:,:],dtype),u3)+torch.mul(gammaconnection_33[0,:,:],diff_u3_y) \
						+ torch.mul(gammaconnection_31[1,:,:],cov_dev_u[0,0,:,:])+torch.mul(gammaconnection_32[1,:,:],cov_dev_u[1,0,:,:])+torch.mul(gammaconnection_33[1,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_yx_3 = torch.unsqueeze(cov_dev_order2_yx_3,dim=0) #[1xHxW]

	cov_dev_order2_yy_3 = diff_u3_yy + torch.mul(derivative_central_y_greylevel(gammaconnection_31[1,:,:],dtype),u1)+torch.mul(gammaconnection_31[1,:,:],diff_u1_y) \
						+ torch.mul(derivative_central_y_greylevel(gammaconnection_32[1,:,:],dtype),u2)+torch.mul(gammaconnection_32[1,:,:],diff_u2_y) \
						+ torch.mul(derivative_central_y_greylevel(gammaconnection_33[1,:,:],dtype),u3)+torch.mul(gammaconnection_33[1,:,:],diff_u3_y) \
						+ torch.mul(gammaconnection_31[1,:,:],cov_dev_u[0,1,:,:])+torch.mul(gammaconnection_32[1,:,:],cov_dev_u[1,1,:,:])+torch.mul(gammaconnection_33[1,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_yy_3 = torch.unsqueeze(cov_dev_order2_yy_3,dim=0) #[1xHxW]

	cov_dev_order2_3 = torch.cat((cov_dev_order2_xx_3,cov_dev_order2_xy_3,cov_dev_order2_yx_3,cov_dev_order2_yy_3),dim=0) #[4xHxW]
	cov_dev_order2_3 = torch.unsqueeze(cov_dev_order2_3,dim=0) #[1x4xHxW]

	cov_dev_order2 = torch.cat((cov_dev_order2_1,cov_dev_order2_2,cov_dev_order2_3),dim=0) #[3x4xHxW]

	return cov_dev_order2 #[3x4xHxW]


def covariant_derivative_second_order_GL3_optimal_implementation2(u, omegaE_11,omegaE_12,omegaE_13,omegaE_21,omegaE_22,omegaE_23,omegaE_31,omegaE_32,omegaE_33,omegaTM_11,omegaTM_12,omegaTM_21,omegaTM_22,dtype):
	
	cov_dev_u = covariant_derivative_first_order_GL3_optimal(u, omegaE_11,omegaE_12,omegaE_13,omegaE_21,omegaE_22,omegaE_23,omegaE_31,omegaE_32,omegaE_33,dtype)
	gammaconnection_11,gammaconnection_12,gammaconnection_13,gammaconnection_21,gammaconnection_22,gammaconnection_23,gammaconnection_31,gammaconnection_32,gammaconnection_33 = connection1form_GL3_optimal(u, omegaE_11,omegaE_12,omegaE_13,omegaE_21,omegaE_22,omegaE_23,omegaE_31,omegaE_32,omegaE_33,dtype)

	u1 = torch.narrow(u,1,0,1)       # [1x1xHxW]
	u1 = torch.squeeze(u1)   		 # [HxW]
	u1dusse = torch.unsqueeze(u1,dim=0)   # [1xHxW]
	u1tilde = u1dusse.repeat(2,1,1)   	 # [2xHxW]
	u2 = torch.narrow(u,1,1,1)       # [1x1xHxW]
	u2 = torch.squeeze(u2)   		 # [HxW]
	u2dusse = torch.unsqueeze(u2,dim=0)   # [1xHxW]
	u2tilde = u2dusse.repeat(2,1,1)   	 # [2xHxW]
	u3 = torch.narrow(u,1,2,1)       # [1x1xHxW]
	u3 = torch.squeeze(u3)   		 # [HxW]
	u3dusse = torch.unsqueeze(u3,dim=0)   # [1xHxW]
	u3tilde = u3dusse.repeat(2,1,1)       # [2xHxW]

	diff_u1_x = derivative_central_x_greylevel(u1,dtype) #[HxW]
	diff_u1_y = derivative_central_x_greylevel(u1,dtype) #[HxW]
	diff_u1tilde_x = torch.unsqueeze(diff_u1_x,dim=0)    #[1xHxW] 
	diff_u1tilde_y = torch.unsqueeze(diff_u1_y,dim=0)    #[1xHxW] 
	diff_u1 = torch.cat((diff_u1tilde_x,diff_u1tilde_y),dim=0)     #[2xHxW]

	diff_u2_x = derivative_central_x_greylevel(u2,dtype) #[HxW]
	diff_u2_y = derivative_central_x_greylevel(u2,dtype) #[HxW]
	diff_u2tilde_x = torch.unsqueeze(diff_u2_x,dim=0)    #[1xHxW] 
	diff_u2tilde_y = torch.unsqueeze(diff_u2_y,dim=0)    #[1xHxW] 
	diff_u2 = torch.cat((diff_u2tilde_x,diff_u2tilde_y),dim=0)     #[2xHxW]

	diff_u3_x = derivative_central_x_greylevel(u3,dtype) #[HxW]
	diff_u3_y = derivative_central_x_greylevel(u3,dtype) #[HxW]
	diff_u3tilde_x = torch.unsqueeze(diff_u3_x,dim=0)    #[1xHxW] 
	diff_u3tilde_y = torch.unsqueeze(diff_u3_y,dim=0)    #[1xHxW] 
	diff_u3 = torch.cat((diff_u3tilde_x,diff_u3tilde_y),dim=0)     #[2xHxW]

	diff_u1_xx = derivative_xx_greylevel(u1,dtype) #[HxW]
	diff_u1_xy = derivative_xy_greylevel(u1,dtype) #[HxW]
	diff_u1_yy = derivative_yy_greylevel(u1,dtype) #[HxW]

	diff_u2_xx = derivative_xx_greylevel(u2,dtype) #[HxW]
	diff_u2_xy = derivative_xy_greylevel(u2,dtype) #[HxW]
	diff_u2_yy = derivative_yy_greylevel(u2,dtype) #[HxW]

	diff_u3_xx = derivative_xx_greylevel(u3,dtype) #[HxW]
	diff_u3_xy = derivative_xy_greylevel(u3,dtype) #[HxW]
	diff_u3_yy = derivative_yy_greylevel(u3,dtype) #[HxW]


	coeffrep_num = - torch.sum( torch.mul(diff_u1+0.5*(torch.mul(omegaE_13-omegaE_31,u3tilde))+0.5*(torch.mul(omegaE_12-omegaE_21,u2tilde)),2.*torch.mul(omegaE_11,u1tilde)+torch.mul(omegaE_12+omegaE_21,u2tilde)+torch.mul(omegaE_13+omegaE_31,u3tilde)),dim=0) \
					- torch.sum( torch.mul(diff_u2+0.5*(torch.mul(omegaE_21-omegaE_12,u1tilde))+0.5*(torch.mul(omegaE_23-omegaE_32,u3tilde)),2.*torch.mul(omegaE_22,u2tilde)+torch.mul(omegaE_12+omegaE_21,u1tilde)+torch.mul(omegaE_23+omegaE_32,u3tilde)),dim=0) \
					- torch.sum( torch.mul(diff_u3+0.5*(torch.mul(omegaE_31-omegaE_13,u1tilde))+0.5*(torch.mul(omegaE_32-omegaE_23,u2tilde)),2.*torch.mul(omegaE_33,u3tilde)+torch.mul(omegaE_23+omegaE_32,u2tilde)+torch.mul(omegaE_13+omegaE_31,u1tilde)),dim=0) #[HxW]	
					

	coeffrep_den = 2.*torch.sum(torch.mul(torch.mul(omegaE_11,u1tilde),torch.mul(omegaE_11,u1tilde)+torch.mul(omegaE_12+omegaE_21,u2tilde)+torch.mul(omegaE_13+omegaE_31,u3tilde)),dim=0) \
					 + 2.*torch.sum(torch.mul(torch.mul(omegaE_22,u2tilde),torch.mul(omegaE_22,u2tilde)+torch.mul(omegaE_23+omegaE_32,u3tilde)+torch.mul(omegaE_12+omegaE_21,u1tilde)),dim=0)\
					 + 2.*torch.sum(torch.mul(torch.mul(omegaE_33,u3tilde),torch.mul(omegaE_33,u3tilde)+torch.mul(omegaE_13+omegaE_31,u1tilde)+torch.mul(omegaE_23+omegaE_32,u2tilde)),dim=0) \
					+ 0.5*torch.sum(torch.square( torch.mul(omegaE_12+omegaE_21,u1tilde)+torch.mul(omegaE_23+omegaE_32,u3tilde)),dim=0) \
					+ 0.5*torch.sum(torch.square( torch.mul(omegaE_12+omegaE_21,u2tilde)+torch.mul(omegaE_13+omegaE_31,u3tilde)),dim=0) \
					+ 0.5*torch.sum(torch.square( torch.mul(omegaE_13+omegaE_31,u1tilde)+torch.mul(omegaE_23+omegaE_32,u2tilde)),dim=0) #[HxW]


	coeff_rep = torch.div(coeffrep_num,0.001+coeffrep_den) #[HxW]
	diff_coeff_rep_x = torch.div(torch.mul(derivative_central_x_greylevel(coeffrep_num,dtype),coeffrep_den) - torch.mul(derivative_central_x_greylevel(coeffrep_den,dtype),coeffrep_num),0.0001+torch.square(coeffrep_den)) 
	diff_coeff_rep_y = torch.div(torch.mul(derivative_central_y_greylevel(coeffrep_num,dtype),coeffrep_den) - torch.mul(derivative_central_y_greylevel(coeffrep_den,dtype),coeffrep_num),0.0001+torch.square(coeffrep_den)) 
	
	coeff_rep_tmp = torch.unsqueeze(coeff_rep,dim=0)
	coeff_rep_tilde = coeff_rep_tmp.repeat(2,1,1) #[2xHxW]
	
	gammaconnection11 = torch.mul(coeff_rep,omegaE_11) # [2xHxW]
	gammaconnection12 = 0.5*(omegaE_12-omegaE_21)+0.5*torch.mul(coeff_rep_tilde,omegaE_12+omegaE_21) # [2xHxW]
	gammaconnection13 = 0.5*(omegaE_13-omegaE_31)+0.5*torch.mul(coeff_rep_tilde,omegaE_13+omegaE_31) # [2xHxW]
	gammaconnection21 = -0.5*(omegaE_12-omegaE_21)+0.5*torch.mul(coeff_rep_tilde,omegaE_12+omegaE_21) # [2xHxW]
	gammaconnection22 = torch.mul(coeff_rep,omegaE_22) # [2xHxW]
	gammaconnection23 = 0.5*(omegaE_23-omegaE_32)+0.5*torch.mul(coeff_rep_tilde,omegaE_23+omegaE_32) # [2xHxW]
	gammaconnection31 = -0.5*(omegaE_13-omegaE_31)+0.5*torch.mul(coeff_rep_tilde,omegaE_13+omegaE_31) # [2xHxW]
	gammaconnection32 = -0.5*(omegaE_32-omegaE_23)+0.5*torch.mul(coeff_rep_tilde,omegaE_23+omegaE_32) # [2xHxW]
	gammaconnection33 = torch.mul(coeff_rep,omegaE_33) # [2xHxW]


	diff_gammaconnection11x_x = torch.mul(diff_coeff_rep_x,omegaE_11[0,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_11[0,:,:],dtype)) 
	diff_gammaconnection11x_y = torch.mul(diff_coeff_rep_y,omegaE_11[0,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_11[0,:,:],dtype)) 
	diff_gammaconnection11y_x = torch.mul(diff_coeff_rep_x,omegaE_11[1,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_11[1,:,:],dtype)) 
	diff_gammaconnection11y_y = torch.mul(diff_coeff_rep_y,omegaE_11[1,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_11[1,:,:],dtype)) 

	diff_gammaconnection12x_x = 0.5*(derivative_central_x_greylevel(omegaE_12[0,:,:]-omegaE_21[0,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_x,omegaE_12[0,:,:]+omegaE_21[0,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_12[0,:,:]+omegaE_21[0,:,:],dtype)))
	diff_gammaconnection12x_y = 0.5*(derivative_central_y_greylevel(omegaE_12[0,:,:]-omegaE_21[0,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_x,omegaE_12[0,:,:]+omegaE_21[0,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_12[0,:,:]+omegaE_21[0,:,:],dtype)))
	diff_gammaconnection12y_x = 0.5*(derivative_central_x_greylevel(omegaE_12[1,:,:]-omegaE_21[1,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_y,omegaE_12[1,:,:]+omegaE_21[1,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_12[1,:,:]+omegaE_21[1,:,:],dtype)))
	diff_gammaconnection12y_y = 0.5*(derivative_central_y_greylevel(omegaE_12[1,:,:]-omegaE_21[1,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_y,omegaE_12[1,:,:]+omegaE_21[1,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_12[1,:,:]+omegaE_21[1,:,:],dtype)))

	diff_gammaconnection13x_x = 0.5*(derivative_central_x_greylevel(omegaE_13[0,:,:]-omegaE_31[0,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_x,omegaE_13[0,:,:]+omegaE_31[0,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_13[0,:,:]+omegaE_31[0,:,:],dtype)))
	diff_gammaconnection13x_y = 0.5*(derivative_central_y_greylevel(omegaE_13[0,:,:]-omegaE_31[0,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_x,omegaE_13[0,:,:]+omegaE_31[0,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_13[0,:,:]+omegaE_31[0,:,:],dtype)))
	diff_gammaconnection13y_x = 0.5*(derivative_central_x_greylevel(omegaE_13[1,:,:]-omegaE_31[1,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_y,omegaE_13[1,:,:]+omegaE_31[1,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_13[1,:,:]+omegaE_31[1,:,:],dtype)))
	diff_gammaconnection13y_y = 0.5*(derivative_central_y_greylevel(omegaE_13[1,:,:]-omegaE_31[1,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_y,omegaE_13[1,:,:]+omegaE_31[1,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_13[1,:,:]+omegaE_31[1,:,:],dtype)))

	diff_gammaconnection21x_x = -0.5*(derivative_central_x_greylevel(omegaE_12[0,:,:]-omegaE_21[0,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_x,omegaE_12[0,:,:]+omegaE_21[0,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_12[0,:,:]+omegaE_21[0,:,:],dtype)))
	diff_gammaconnection21x_y = -0.5*(derivative_central_y_greylevel(omegaE_12[0,:,:]-omegaE_21[0,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_x,omegaE_12[0,:,:]+omegaE_21[0,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_12[0,:,:]+omegaE_21[0,:,:],dtype)))
	diff_gammaconnection21y_x = -0.5*(derivative_central_x_greylevel(omegaE_12[1,:,:]-omegaE_21[1,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_y,omegaE_12[1,:,:]+omegaE_21[1,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_12[1,:,:]+omegaE_21[1,:,:],dtype)))
	diff_gammaconnection21y_y = -0.5*(derivative_central_y_greylevel(omegaE_12[1,:,:]-omegaE_21[1,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_y,omegaE_12[1,:,:]+omegaE_21[1,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_12[1,:,:]+omegaE_21[1,:,:],dtype)))

	diff_gammaconnection22x_x = torch.mul(diff_coeff_rep_x,omegaE_22[0,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_22[0,:,:],dtype)) 
	diff_gammaconnection22x_y = torch.mul(diff_coeff_rep_y,omegaE_22[0,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_22[0,:,:],dtype)) 
	diff_gammaconnection22y_x = torch.mul(diff_coeff_rep_x,omegaE_22[1,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_22[1,:,:],dtype)) 
	diff_gammaconnection22y_y = torch.mul(diff_coeff_rep_y,omegaE_22[1,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_22[1,:,:],dtype)) 

	diff_gammaconnection23x_x = 0.5*(derivative_central_x_greylevel(omegaE_23[0,:,:]-omegaE_32[0,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_x,omegaE_23[0,:,:]+omegaE_32[0,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_23[0,:,:]+omegaE_32[0,:,:],dtype)))
	diff_gammaconnection23x_y = 0.5*(derivative_central_y_greylevel(omegaE_23[0,:,:]-omegaE_32[0,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_x,omegaE_23[0,:,:]+omegaE_32[0,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_23[0,:,:]+omegaE_32[0,:,:],dtype)))
	diff_gammaconnection23y_x = 0.5*(derivative_central_x_greylevel(omegaE_23[1,:,:]-omegaE_32[1,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_y,omegaE_23[1,:,:]+omegaE_32[1,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_23[1,:,:]+omegaE_32[1,:,:],dtype)))
	diff_gammaconnection23y_y = 0.5*(derivative_central_y_greylevel(omegaE_23[1,:,:]-omegaE_32[1,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_y,omegaE_23[1,:,:]+omegaE_32[1,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_23[1,:,:]+omegaE_32[1,:,:],dtype)))

	diff_gammaconnection31x_x = -0.5*(derivative_central_x_greylevel(omegaE_13[0,:,:]-omegaE_31[0,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_x,omegaE_13[0,:,:]+omegaE_31[0,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_13[0,:,:]+omegaE_31[0,:,:],dtype)))
	diff_gammaconnection31x_y = -0.5*(derivative_central_y_greylevel(omegaE_13[0,:,:]-omegaE_31[0,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_x,omegaE_13[0,:,:]+omegaE_31[0,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_13[0,:,:]+omegaE_31[0,:,:],dtype)))
	diff_gammaconnection31y_x = -0.5*(derivative_central_x_greylevel(omegaE_13[1,:,:]-omegaE_31[1,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_y,omegaE_13[1,:,:]+omegaE_31[1,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_13[1,:,:]+omegaE_31[1,:,:],dtype)))
	diff_gammaconnection31y_y = -0.5*(derivative_central_y_greylevel(omegaE_13[1,:,:]-omegaE_31[1,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_y,omegaE_13[1,:,:]+omegaE_31[1,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_13[1,:,:]+omegaE_31[1,:,:],dtype)))

	diff_gammaconnection32x_x = -0.5*(derivative_central_x_greylevel(omegaE_23[0,:,:]-omegaE_32[0,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_x,omegaE_23[0,:,:]+omegaE_32[0,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_23[0,:,:]+omegaE_32[0,:,:],dtype)))
	diff_gammaconnection32x_y = -0.5*(derivative_central_y_greylevel(omegaE_23[0,:,:]-omegaE_32[0,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_x,omegaE_23[0,:,:]+omegaE_32[0,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_23[0,:,:]+omegaE_32[0,:,:],dtype)))
	diff_gammaconnection32y_x = -0.5*(derivative_central_x_greylevel(omegaE_23[1,:,:]-omegaE_32[1,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_y,omegaE_23[1,:,:]+omegaE_32[1,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_23[1,:,:]+omegaE_32[1,:,:],dtype)))
	diff_gammaconnection32y_y = -0.5*(derivative_central_y_greylevel(omegaE_23[1,:,:]-omegaE_32[1,:,:],dtype))+0.5*(torch.mul(diff_coeff_rep_y,omegaE_23[1,:,:]+omegaE_32[1,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_23[1,:,:]+omegaE_32[1,:,:],dtype)))

	diff_gammaconnection33x_x = torch.mul(diff_coeff_rep_x,omegaE_33[0,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_33[0,:,:],dtype)) 
	diff_gammaconnection33x_y = torch.mul(diff_coeff_rep_y,omegaE_33[0,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_33[0,:,:],dtype)) 
	diff_gammaconnection33y_x = torch.mul(diff_coeff_rep_x,omegaE_33[1,:,:])+torch.mul(coeff_rep,derivative_central_x_greylevel(omegaE_33[1,:,:],dtype)) 
	diff_gammaconnection33y_y = torch.mul(diff_coeff_rep_y,omegaE_33[1,:,:])+torch.mul(coeff_rep,derivative_central_y_greylevel(omegaE_33[1,:,:],dtype)) 


	cov_dev_order2_xx_1 = diff_u1_xx + torch.mul(diff_gammaconnection11x_x,u1)+torch.mul(gammaconnection_11[0,:,:],diff_u1_x) \
						+ torch.mul(diff_gammaconnection12x_x,u2)+torch.mul(gammaconnection_12[0,:,:],diff_u2_x) \
						+ torch.mul(diff_gammaconnection13x_x,u3)+torch.mul(gammaconnection_13[0,:,:],diff_u3_x) \
						+ torch.mul(gammaconnection_11[0,:,:],cov_dev_u[0,0,:,:])+torch.mul(gammaconnection_12[0,:,:],cov_dev_u[1,0,:,:])+torch.mul(gammaconnection_13[0,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_xx_1 = torch.unsqueeze(cov_dev_order2_xx_1,dim=0) #[1xHxW]

	cov_dev_order2_xy_1 = diff_u1_xy + torch.mul(diff_gammaconnection11y_x,u1)+torch.mul(gammaconnection_11[1,:,:],diff_u1_x) \
						+ torch.mul(diff_gammaconnection12y_x,u2)+torch.mul(gammaconnection_12[1,:,:],diff_u2_x) \
						+ torch.mul(diff_gammaconnection13y_x,u3)+torch.mul(gammaconnection_13[1,:,:],diff_u3_x) \
						+ torch.mul(gammaconnection_11[0,:,:],cov_dev_u[0,1,:,:])+torch.mul(gammaconnection_12[0,:,:],cov_dev_u[1,1,:,:])+torch.mul(gammaconnection_13[0,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_xy_1 = torch.unsqueeze(cov_dev_order2_xy_1,dim=0) #[1xHxW]

	cov_dev_order2_yx_1 = diff_u1_xy + torch.mul(diff_gammaconnection11x_y,u1)+torch.mul(gammaconnection_11[0,:,:],diff_u1_y) \
						+ torch.mul(diff_gammaconnection12x_y,u2)+torch.mul(gammaconnection_12[0,:,:],diff_u2_y) \
						+ torch.mul(diff_gammaconnection13x_y,u3)+torch.mul(gammaconnection_13[0,:,:],diff_u3_y) \
						+ torch.mul(gammaconnection_11[1,:,:],cov_dev_u[0,0,:,:])+torch.mul(gammaconnection_12[1,:,:],cov_dev_u[1,0,:,:])+torch.mul(gammaconnection_13[1,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_yx_1 = torch.unsqueeze(cov_dev_order2_yx_1,dim=0) #[1xHxW]

	cov_dev_order2_yy_1 = diff_u1_yy + torch.mul(diff_gammaconnection11y_y,u1)+torch.mul(gammaconnection_11[1,:,:],diff_u1_y) \
						+ torch.mul(diff_gammaconnection12y_y,u2)+torch.mul(gammaconnection_12[1,:,:],diff_u2_y) \
						+ torch.mul(diff_gammaconnection13y_y,u3)+torch.mul(gammaconnection_13[1,:,:],diff_u3_y) \
						+ torch.mul(gammaconnection_11[1,:,:],cov_dev_u[0,1,:,:])+torch.mul(gammaconnection_12[1,:,:],cov_dev_u[1,1,:,:])+torch.mul(gammaconnection_13[1,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_yy_1 = torch.unsqueeze(cov_dev_order2_yy_1,dim=0) #[1xHxW]

	cov_dev_order2_1 = torch.cat((cov_dev_order2_xx_1,cov_dev_order2_xy_1,cov_dev_order2_yx_1,cov_dev_order2_yy_1),dim=0) #[4xHxW]
	cov_dev_order2_1 = torch.unsqueeze(cov_dev_order2_1,dim=0) #[1x4xHxW]


	cov_dev_order2_xx_2 = diff_u2_xx + torch.mul(diff_gammaconnection21x_x,u1)+torch.mul(gammaconnection_21[0,:,:],diff_u1_x) \
						+ torch.mul(diff_gammaconnection22x_x,u2)+torch.mul(gammaconnection_22[0,:,:],diff_u2_x) \
						+ torch.mul(diff_gammaconnection23x_x,u3)+torch.mul(gammaconnection_23[0,:,:],diff_u3_x) \
						+ torch.mul(gammaconnection_21[0,:,:],cov_dev_u[0,0,:,:])+torch.mul(gammaconnection_22[0,:,:],cov_dev_u[1,0,:,:])+torch.mul(gammaconnection_23[0,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_xx_2 = torch.unsqueeze(cov_dev_order2_xx_2,dim=0) #[1xHxW]

	cov_dev_order2_xy_2 = diff_u2_xy + torch.mul(diff_gammaconnection21y_x,u1)+torch.mul(gammaconnection_21[1,:,:],diff_u1_x) \
						+ torch.mul(diff_gammaconnection22y_x,u2)+torch.mul(gammaconnection_22[1,:,:],diff_u2_x) \
						+ torch.mul(diff_gammaconnection23y_x,u3)+torch.mul(gammaconnection_23[1,:,:],diff_u3_x) \
						+ torch.mul(gammaconnection_21[0,:,:],cov_dev_u[0,1,:,:])+torch.mul(gammaconnection_22[0,:,:],cov_dev_u[1,1,:,:])+torch.mul(gammaconnection_23[0,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_xy_2 = torch.unsqueeze(cov_dev_order2_xy_2,dim=0) #[1xHxW]

	cov_dev_order2_yx_2 = diff_u2_xy + torch.mul(diff_gammaconnection21x_y,u1)+torch.mul(gammaconnection_21[0,:,:],diff_u1_y) \
						+ torch.mul(diff_gammaconnection22x_y,u2)+torch.mul(gammaconnection_22[0,:,:],diff_u2_y) \
						+ torch.mul(diff_gammaconnection23x_y,u3)+torch.mul(gammaconnection_23[0,:,:],diff_u3_y) \
						+ torch.mul(gammaconnection_21[1,:,:],cov_dev_u[0,0,:,:])+torch.mul(gammaconnection_22[1,:,:],cov_dev_u[1,0,:,:])+torch.mul(gammaconnection_23[1,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_yx_2 = torch.unsqueeze(cov_dev_order2_yx_2,dim=0) #[1xHxW]

	cov_dev_order2_yy_2 = diff_u2_yy + torch.mul(diff_gammaconnection21y_y,u1)+torch.mul(gammaconnection_21[1,:,:],diff_u1_y) \
						+ torch.mul(diff_gammaconnection22y_y,u2)+torch.mul(gammaconnection_22[1,:,:],diff_u2_y) \
						+ torch.mul(diff_gammaconnection23y_y,u3)+torch.mul(gammaconnection_23[1,:,:],diff_u3_y) \
						+ torch.mul(gammaconnection_21[1,:,:],cov_dev_u[0,1,:,:])+torch.mul(gammaconnection_22[1,:,:],cov_dev_u[1,1,:,:])+torch.mul(gammaconnection_23[1,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_yy_2 = torch.unsqueeze(cov_dev_order2_yy_2,dim=0) #[1xHxW]

	cov_dev_order2_2 = torch.cat((cov_dev_order2_xx_2,cov_dev_order2_xy_2,cov_dev_order2_yx_2,cov_dev_order2_yy_2),dim=0) #[4xHxW]
	cov_dev_order2_2 = torch.unsqueeze(cov_dev_order2_2,dim=0) #[1x4xHxW]

	cov_dev_order2_xx_3 = diff_u3_xx + torch.mul(diff_gammaconnection31x_x,u1)+torch.mul(gammaconnection_31[0,:,:],diff_u1_x) \
						+ torch.mul(diff_gammaconnection32x_x,u2)+torch.mul(gammaconnection_32[0,:,:],diff_u2_x) \
						+ torch.mul(diff_gammaconnection33x_x,u3)+torch.mul(gammaconnection_33[0,:,:],diff_u3_x) \
						+ torch.mul(gammaconnection_31[0,:,:],cov_dev_u[0,0,:,:])+torch.mul(gammaconnection_32[0,:,:],cov_dev_u[1,0,:,:])+torch.mul(gammaconnection_33[0,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_xx_3 = torch.unsqueeze(cov_dev_order2_xx_3,dim=0) #[1xHxW]

	cov_dev_order2_xy_3 = diff_u3_xy + torch.mul(diff_gammaconnection31y_x,u1)+torch.mul(gammaconnection_31[1,:,:],diff_u1_x) \
						+ torch.mul(diff_gammaconnection32y_x,u2)+torch.mul(gammaconnection_32[1,:,:],diff_u2_x) \
						+ torch.mul(diff_gammaconnection33y_x,u3)+torch.mul(gammaconnection_33[1,:,:],diff_u3_x) \
						+ torch.mul(gammaconnection_31[0,:,:],cov_dev_u[0,1,:,:])+torch.mul(gammaconnection_32[0,:,:],cov_dev_u[1,1,:,:])+torch.mul(gammaconnection_33[0,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_xy_3 = torch.unsqueeze(cov_dev_order2_xy_3,dim=0) #[1xHxW]

	cov_dev_order2_yx_3 = diff_u3_xy + torch.mul(diff_gammaconnection31x_y,u1)+torch.mul(gammaconnection_31[0,:,:],diff_u1_y) \
						+ torch.mul(diff_gammaconnection32x_y,u2)+torch.mul(gammaconnection_32[0,:,:],diff_u2_y) \
						+ torch.mul(diff_gammaconnection33x_y,u3)+torch.mul(gammaconnection_33[0,:,:],diff_u3_y) \
						+ torch.mul(gammaconnection_31[1,:,:],cov_dev_u[0,0,:,:])+torch.mul(gammaconnection_32[1,:,:],cov_dev_u[1,0,:,:])+torch.mul(gammaconnection_33[1,:,:],cov_dev_u[2,0,:,:])
	cov_dev_order2_yx_3 = torch.unsqueeze(cov_dev_order2_yx_3,dim=0) #[1xHxW]

	cov_dev_order2_yy_3 = diff_u3_yy + torch.mul(diff_gammaconnection31y_y,u1)+torch.mul(gammaconnection_31[1,:,:],diff_u1_y) \
						+ torch.mul(diff_gammaconnection32y_y,u2)+torch.mul(gammaconnection_32[1,:,:],diff_u2_y) \
						+ torch.mul(diff_gammaconnection33y_y,u3)+torch.mul(gammaconnection_33[1,:,:],diff_u3_y) \
						+ torch.mul(gammaconnection_31[1,:,:],cov_dev_u[0,1,:,:])+torch.mul(gammaconnection_32[1,:,:],cov_dev_u[1,1,:,:])+torch.mul(gammaconnection_33[1,:,:],cov_dev_u[2,1,:,:])
	cov_dev_order2_yy_3 = torch.unsqueeze(cov_dev_order2_yy_3,dim=0) #[1xHxW]

	cov_dev_order2_3 = torch.cat((cov_dev_order2_xx_3,cov_dev_order2_xy_3,cov_dev_order2_yx_3,cov_dev_order2_yy_3),dim=0) #[4xHxW]
	cov_dev_order2_3 = torch.unsqueeze(cov_dev_order2_3,dim=0) #[1x4xHxW]

	cov_dev_order2 = torch.cat((cov_dev_order2_1,cov_dev_order2_2,cov_dev_order2_3),dim=0) #[3x4xHxW]

	return cov_dev_order2 #[3x4xHxW]



def norm_covariant_derivative_second_order_GL3_optimal(u,epsilon, omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,omegaTM_11,omegaTM_12,omegaTM_21,omegaTM_22,invg11,invg12,invg22,dtype):

	cov_dev_order2 = covariant_derivative_second_order_GL3_optimal(u, omegaE_11,omegaE_12,omegaE_13,omegaE_21,omegaE_22,omegaE_23,omegaE_31,omegaE_32,omegaE_33,omegaTM_11,omegaTM_12,omegaTM_21,omegaTM_22,dtype) #[3x4xHxW]

	norm_cov_dev_order2_squared = torch.mul(torch.square(invg11), torch.sum(torch.square(cov_dev_order2[:,0,:,:]),dim=0)) \
								+ torch.mul(torch.square(invg22), torch.sum(torch.square(cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.square(invg12), torch.sum(torch.mul(cov_dev_order2[:,0,:,:],cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg22),torch.sum(torch.mul(cov_dev_order2[:,1,:,:],cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg22),torch.sum(torch.mul(cov_dev_order2[:,2,:,:],cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg11),torch.sum(torch.mul(cov_dev_order2[:,1,:,:],cov_dev_order2[:,0,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg11),torch.sum(torch.mul(cov_dev_order2[:,2,:,:],cov_dev_order2[:,0,:,:]),dim=0)) \
								+ torch.mul(torch.mul(invg11,invg22),torch.sum(torch.square(cov_dev_order2[:,1,:,:]),dim=0)) \
								+ torch.mul(torch.mul(invg11,invg22),torch.sum(torch.square(cov_dev_order2[:,2,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.square(invg12),torch.sum(torch.mul(cov_dev_order2[:,1,:,:],cov_dev_order2[:,2,:,:]),dim=0))

	norm_cov_dev_order2 = torch.sqrt(epsilon+norm_cov_dev_order2_squared) #[HxW]

	return norm_cov_dev_order2		

def norm_covariant_derivative_second_order_GL3_optimal_implementation2(u,epsilon, omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,omegaTM_11,omegaTM_12,omegaTM_21,omegaTM_22,invg11,invg12,invg22,dtype):

	cov_dev_order2 = covariant_derivative_second_order_GL3_optimal_implementation2(u, omegaE_11,omegaE_12,omegaE_13,omegaE_21,omegaE_22,omegaE_23,omegaE_31,omegaE_32,omegaE_33,omegaTM_11,omegaTM_12,omegaTM_21,omegaTM_22,dtype) #[3x4xHxW]

	norm_cov_dev_order2_squared = torch.mul(torch.square(invg11), torch.sum(torch.square(cov_dev_order2[:,0,:,:]),dim=0)) \
								+ torch.mul(torch.square(invg22), torch.sum(torch.square(cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.square(invg12), torch.sum(torch.mul(cov_dev_order2[:,0,:,:],cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg22),torch.sum(torch.mul(cov_dev_order2[:,1,:,:],cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg22),torch.sum(torch.mul(cov_dev_order2[:,2,:,:],cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg11),torch.sum(torch.mul(cov_dev_order2[:,1,:,:],cov_dev_order2[:,0,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg11),torch.sum(torch.mul(cov_dev_order2[:,2,:,:],cov_dev_order2[:,0,:,:]),dim=0)) \
								+ torch.mul(torch.mul(invg11,invg22),torch.sum(torch.square(cov_dev_order2[:,1,:,:]),dim=0)) \
								+ torch.mul(torch.mul(invg11,invg22),torch.sum(torch.square(cov_dev_order2[:,2,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.square(invg12),torch.sum(torch.mul(cov_dev_order2[:,1,:,:],cov_dev_order2[:,2,:,:]),dim=0))

	norm_cov_dev_order2 = torch.sqrt(epsilon+norm_cov_dev_order2_squared) #[HxW]

	return norm_cov_dev_order2	


def norm_covariant_derivative_second_order_GL3(u,epsilon, red, green, blue, diff_red_x,diff_green_x,diff_blue_x, diff_red_y,diff_green_y,diff_blue_y,diff_red_xx,diff_red_xy,diff_red_yy,diff_green_xx,diff_green_xy,diff_green_yy,diff_blue_xx,diff_blue_xy,diff_blue_yy, omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,omegaTM_11,omegaTM_12,omegaTM_21,omegaTM_22,invg11,invg12,invg22,dtype):

	cov_dev_order2 = covariant_derivative_second_order_GL3(u,red, green, blue, diff_red_x,diff_green_x,diff_blue_x, diff_red_y,diff_green_y,diff_blue_y,diff_red_xx,diff_red_xy,diff_red_yy,diff_green_xx,diff_green_xy,diff_green_yy,diff_blue_xx,diff_blue_xy,diff_blue_yy, omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,omegaTM_11,omegaTM_12,omegaTM_21,omegaTM_22,dtype) #[3x4xHxW]

	norm_cov_dev_order2_squared = torch.mul(torch.square(invg11), torch.sum(torch.square(cov_dev_order2[:,0,:,:]),dim=0)) \
								+ torch.mul(torch.square(invg22), torch.sum(torch.square(cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.square(invg12), torch.sum(torch.mul(cov_dev_order2[:,0,:,:],cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg22),torch.sum(torch.mul(cov_dev_order2[:,1,:,:],cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg22),torch.sum(torch.mul(cov_dev_order2[:,2,:,:],cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg11),torch.sum(torch.mul(cov_dev_order2[:,1,:,:],cov_dev_order2[:,0,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg11),torch.sum(torch.mul(cov_dev_order2[:,2,:,:],cov_dev_order2[:,0,:,:]),dim=0)) \
								+ torch.mul(torch.mul(invg11,invg22),torch.sum(torch.square(cov_dev_order2[:,1,:,:]),dim=0)) \
								+ torch.mul(torch.mul(invg11,invg22),torch.sum(torch.square(cov_dev_order2[:,2,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.square(invg12),torch.sum(torch.mul(cov_dev_order2[:,1,:,:],cov_dev_order2[:,2,:,:]),dim=0))

	norm_cov_dev_order2 = torch.sqrt(epsilon+norm_cov_dev_order2_squared) #[HxW]

	return norm_cov_dev_order2		



def norm_covariant_derivative_second_order_general(u,epsilon, omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,omegaTM_11,omegaTM_12,omegaTM_21,omegaTM_22,invg11,invg12,invg22,dtype):

	cov_dev_order2 = covariant_derivative_second_order_general(u,omegaE_11, omegaE_12, omegaE_13, omegaE_21, omegaE_22, omegaE_23, omegaE_31, omegaE_32, omegaE_33,omegaTM_11,omegaTM_12,omegaTM_21,omegaTM_22,dtype) #[3x4xHxW]

	norm_cov_dev_order2_squared = torch.mul(torch.square(invg11), torch.sum(torch.square(cov_dev_order2[:,0,:,:]),dim=0)) \
								+ torch.mul(torch.square(invg22), torch.sum(torch.square(cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.square(invg12), torch.sum(torch.mul(cov_dev_order2[:,0,:,:],cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg22),torch.sum(torch.mul(cov_dev_order2[:,1,:,:],cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg22),torch.sum(torch.mul(cov_dev_order2[:,2,:,:],cov_dev_order2[:,3,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg11),torch.sum(torch.mul(cov_dev_order2[:,1,:,:],cov_dev_order2[:,0,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.mul(invg12,invg11),torch.sum(torch.mul(cov_dev_order2[:,2,:,:],cov_dev_order2[:,0,:,:]),dim=0)) \
								+ torch.mul(torch.mul(invg11,invg22),torch.sum(torch.square(cov_dev_order2[:,1,:,:]),dim=0)) \
								+ torch.mul(torch.mul(invg11,invg22),torch.sum(torch.square(cov_dev_order2[:,2,:,:]),dim=0)) \
								+ 2.*torch.mul(torch.square(invg12),torch.sum(torch.mul(cov_dev_order2[:,1,:,:],cov_dev_order2[:,2,:,:]),dim=0))

	norm_cov_dev_order2 = torch.sqrt(epsilon+norm_cov_dev_order2_squared) #[HxW]

	return norm_cov_dev_order2							



	






'''def norm_first_order_covariant_derivative_greylevel(u,epsilon,beta,dtype,invg11,invg12,invg22):

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

	return norm_third_order_covariant_derivative'''






