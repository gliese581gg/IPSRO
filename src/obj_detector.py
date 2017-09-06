#!/usr/bin/env python
# -*- encoding: UTF-8 -*-
from darkflow.net.build import TFNet
import numpy as np
import cv2
import time
import sys
import argparse
import os

#ROS modules
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion , Twist, Pose, PoseStamped, Vector3
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32,String
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from tf import TransformListener, Transformer, transformations
from std_srvs.srv import Empty
from dip_jychoi.msg import objs, objs_array
import DIP_load_config
import tensorflow
from threading import Thread


class obj_detector:
	options = {"model": "cfg/yolo.cfg", "load": "object_model/yolo.weights", "threshold": 0.3 ,"gpu":0.2 , "summary":None}
	
		
	def __init__(self,params):
		
		self.params = params
		
		self.match_images = {}
		
		ld = os.listdir('./object_images/')
		
		for fn in ld :
			if fn.split('.')[-1] != 'png' or len(fn.split('_'))!=3 : continue
			on = fn.split('.')[0]
			img = cv2.imread('./object_images/'+fn)
			self.match_images[on] = img
		
		print 'matching objects loaded : ' , self.match_images.keys()

		self.point_cloud = None

		
		self.object_id = 0
		self.object_id_max = 999999
		self.msg_idx = 0
		self.msg_idx_max = 9999999	
		
		#Darknet
		self.gg = tensorflow.Graph()		
		with self.gg.as_default() as g:		
			self.tfnet = TFNet(self.options)
		self.classes = open('cfg/coco.names','r').readlines()
		
		if self.params['show_od'] : 
			cv2.startWindowThread()
			cv2.namedWindow('Object_detector')
		
		#ROS
		self.cvbridge = CvBridge()		
		
		self.transform = TransformListener()
		self.transformer = Transformer(True,rospy.Duration(10.0))

		self.RGB_TOPIC = params['rgb_topic']
		self.Depth_TOPIC = params['depth_topic']
		self.OBJ_TOPIC = params['obj_topic']
		
		self.depth_sub = rospy.Subscriber(self.Depth_TOPIC, Image, self.callback_depth, queue_size=1)
		
		time.sleep(1)
		
		self.rgb_sub = rospy.Subscriber(self.RGB_TOPIC, Image, self.callback_image, queue_size=1)
		self.obj_pub = rospy.Publisher(self.OBJ_TOPIC,objs_array,queue_size=1)

		self.tttt =time.time()
		time.sleep(1)
		print ('[DIP]  Object Detector Module Ready!')

	
		
	
	def compare_hist (self,img1,img2): 
		img1 = cv2.resize(img1,(32,32))
		img2 = cv2.resize(img2,(32,32))
		img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
		img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)

		hist0 = cv2.calcHist([img1],[0, 1], None, [10, 16], [0,180,0,256])
		hist1 = cv2.calcHist([img2],[0, 1], None, [10, 16], [0,180,0,256])
		score = cv2.compareHist(hist0,hist1,0) #method 0~6
		#print score
		return score

		
	def callback_depth(self,msg): #return point cloud from depth image
		img = self.cvbridge.imgmsg_to_cv2(msg, 'passthrough')
		
		# Create point cloud from depth (for speed up)
		
		offset_x = -120
		offset_y = -160
		x_temp = -(np.tile(np.arange(img.shape[0]).reshape(img.shape[0],1) , (1,img.shape[1])) + offset_x)
		y_temp = -(np.tile(np.arange(img.shape[1]).reshape(1,img.shape[1]) , (img.shape[0],1)) + offset_y)
		
		fx = 525.0 * 0.54 ; fy = 525.0 * 0.54 ; 

		z = img/1000.0

		x = (x_temp) * z / fx
		y = (y_temp) * z / fy

		cloud_np = np.zeros( ( img.shape[0] , img.shape[1] , 3 ) )	

		cloud_np[:,:,0] = z
		cloud_np[:,:,1] = y
		cloud_np[:,:,2] = x
		
		self.point_cloud = cloud_np.copy()
		
			
	def callback_image(self,msg):
		if self.point_cloud is None : 
			print( "No point cloud data" )
			return None
		tic = time.time()
		img = self.cvbridge.imgmsg_to_cv2(msg, 'bgr8')
		img = cv2.resize(img, (320,240) )
		detections = self.tfnet.return_predict(img)
		img_display = img.copy()

		objarray = objs_array()
		objarray.comm_delay = time.time()-self.tttt
		print '[DIP]  Detection. time elapsed : ' ,time.time()-self.tttt
		self.tttt = time.time()
		objarray.header = msg.header
		objarray.header.stamp = rospy.Time.from_sec(time.time())
		objarray.msg_idx = self.msg_idx
		self.msg_idx += 1
		if self.msg_idx > self.msg_idx_max : self.msg_idx = 0		
		temp = []
		objarray.header.frame_id = msg.header.frame_id
		
		temp_tt = 0
		
		for i in range(len(detections)):
			obj = objs()
			obj.object_id = self.object_id
			self.object_id += 1
			if self.object_id > self.object_id_max : self.object_id = 0
			obj.person_id = -1 #unknown
			obj.person_name = ''
			obj.class_string = detections[i]['label']
			obj.tags.append(detections[i]['label'])
			if obj.class_string == 'person' : obj.tags.append('people')
			tlx = int(detections[i]['topleft']['y'])
			tly = int(detections[i]['topleft']['x'])
			brx = int(detections[i]['bottomright']['y'])
			bry = int(detections[i]['bottomright']['x'])
			
			x = (tlx + brx)/2
			y = (tly + bry)/2
			h = (brx - tlx)/2
			w = (bry - tly)/2

			obj.x = x
			obj.y = y
			obj.h = h
			obj.w = w
			obj.confidence = detections[i]['confidence']
			
			crop = img[ max(0,x-h) : min(img.shape[0],x+h) , max(0,y-w) : min(img.shape[1],y+w) ]
			
			ttiicc = time.time()
			max_score = -1
			sub_class = None
			for mi in self.match_images.keys() :
				mi_spl = mi.split('_')
				mi_cls = mi_spl[0]
				mi_subcls = mi_spl[1]
				mi_idx = mi_spl[2]
				ob_cls = obj.class_string
				if mi_cls in self.class_reroute.keys():
					mi_cls = self.class_reroute[mi_cls]
				if ob_cls in self.class_reroute.keys():
					ob_cls = self.class_reroute[ob_cls]
				if ob_cls != mi_cls : continue
				scr = self.compare_hist(crop,self.match_images[mi])
				#print mi, scr,
				if max_score < scr : 
					max_score = scr
					sub_class = mi_subcls
			#print ''
			temp_tt += time.time()-ttiicc
			if sub_class is not None : obj.tags.append(sub_class)		
				
			if self.params['show_od']:
				cv2.rectangle(img_display,(tly,tlx),(bry,brx),(0,255,0),2)
				lbl = detections[i]['label'] if sub_class is None else sub_class
				cv2.putText(img_display,lbl,(tly,tlx-8),cv2.FONT_HERSHEY_SIMPLEX,0.3,color=(0,0,0),thickness=1)
			
			obj.cropped = self.cvbridge.cv2_to_imgmsg(crop,"bgr8")			
			cropped_point = self.point_cloud[obj.x-obj.h : obj.x+obj.h , obj.y-obj.w : obj.y+obj.w ]
			obj.cropped_cloud = self.cvbridge.cv2_to_imgmsg(cropped_point,encoding="passthrough") 

			point_x = min( max(0, int(obj.x - 0.5*obj.h) ) , 240 )
			
			
			if self.params['use_loc'] : 
			
				pose_wrt_robot = self.get_pos_wrt_robot(point_x,obj.y,scan_len=obj.h,scan='point')
				if (pose_wrt_robot == 0).all() : continue
				if pose_wrt_robot[0] > 8.0 : continue #max range = 10m??
				obj.pose_wrt_robot.position.x = pose_wrt_robot[0]
				obj.pose_wrt_robot.position.y = pose_wrt_robot[1]
				obj.pose_wrt_robot.position.z = pose_wrt_robot[2]
				pose_wrt_map = self.get_loc(pose_wrt_robot)[0]
				obj.pose_wrt_map.position.x = pose_wrt_map[0]
				obj.pose_wrt_map.position.y = pose_wrt_map[1]
				obj.pose_wrt_map.position.z = pose_wrt_map[2]
				pose_wrt_odom = self.get_loc(pose_wrt_robot,target='odom')[0]
				obj.pose_wrt_odom.position.x = pose_wrt_odom[0]
				obj.pose_wrt_odom.position.y = pose_wrt_odom[1]
				obj.pose_wrt_odom.position.z = pose_wrt_odom[2]
				obj.valid_pose = 1
			
			temp.append(  obj   )
		
		#print temp_tt
		objarray.objects = temp
		objarray.scene_rgb = msg
		objarray.scene_cloud = self.cvbridge.cv2_to_imgmsg(self.point_cloud,'passthrough')
		
		if self.params['show_od']:
			cv2.imshow('Object_detector',cv2.resize(img_display,(640,480)))
	
		self.obj_pub.publish(objarray)
		#print 'detection_process : ' , time.time()-tic
	

	def get_pos_wrt_robot(self,x,y,size=10,scan_len=50,scan='point'):
		#scan : point(around), vertical(line)
		if scan == 'point':
			x1 = min(240, max(0, x - size//2) )
			x2 = min(240, max(0, x + size//2) )
			y1 = min(320, max(0, y - size//2) )
			y2 = min(320, max(0, y + size//2) )

			roi = self.point_cloud[x1:x2,y1:y2]
			mask = roi[:,:,0]>0
			masked = roi[mask]
			if masked.size == 0 : return np.array([0,0,0])
			mask = masked[:,0]==masked[:,0].min()
			masked = masked[mask]
			return masked[0]#self.point_cloud[x,y]
		else :
			xx1 = min(240,max(0,x-scan_len))
			xx2 = min(240,max(0,x+scan_len))

			roi = self.point_cloud[xx1:xx2,y-2:y+2,:]
			mask = roi[:,:,0]>0
			masked = roi[mask]
			if masked.size == 0 : return np.array([0,0,0])
			mask = masked[:,0]==masked[:,0].min()
			masked = masked[mask]
			return masked[0]#self.point_cloud[x,y]
		
	def get_loc(self,p=np.array([0,0,0]),o=np.array([0,0,0,1]),source='CameraTop_frame',target='map'):#pose = np.array([x,y,z]) : position w.r.t. robot
		pp = PoseStamped()
		pp.pose.position.x = p[0]
		pp.pose.position.y = p[1]
		pp.pose.position.z = p[2]
		pp.pose.orientation.x = o[0]
		pp.pose.orientation.y = o[1]
		pp.pose.orientation.z = o[2]
		pp.pose.orientation.w = o[3]
		#pp.header.stamp = rospy.get_rostime()
		pp.header.frame_id = source #'CameraDepth_frame'
		self.transform.waitForTransform(target,source,time=rospy.Time(),timeout=rospy.Duration(3.0))
		asdf = self.transform.getLatestCommonTime(target,source)
		pp.header.stamp = asdf

		result = self.transform.transformPose(target,pp)
		result_p = np.array([result.pose.position.x,result.pose.position.y,result.pose.position.z])
		result_o = np.array([result.pose.orientation.x,result.pose.orientation.y,result.pose.orientation.z,result.pose.orientation.w])
		return result_p, result_o
	
def main():
	params = DIP_load_config.load_config()	
	rospy.init_node("object_detector")	
	yolo = obj_detector(params)
	rospy.spin()


if __name__=='__main__':	
	main()
