
def load_config():
	
	config = {}

	f = open('DIP_config.txt','r').readlines()

	for l in f :
		if len(l.strip()) < 1 : continue
		if l.strip()[0] == '#' : continue
		if len(  l.strip().split(':')	) != 2 : continue
		name = l.strip().split(':')[0].strip()	
		value = l.strip().split(':')[1].strip()	
		if value == 'True' : value = True
		if value == 'False' : value = False

		config[name] = value
	'''
	#Main
	'show_integrated_perception' : True,
	#Obj Detector
	'rgb_topic' : 'pepper_robot/camera/front/image_raw',
	'depth_topic' : 'pepper_robot/camera/depth/image_raw',
	'obj_topic' : 'DIP/objects',
	'use_loc' : True,
	'show_od' : True,
	#Reid
	'reid_target_topic' : 'DIP/reid_targets',
	'reid_topic' : 'DIP/people_identified',
	'reid_thr' : 0.75,
	#Pose
	'pose_topic' : 'DIP/people_w_pose',
	#Captioning
	'captioning_topic' : 'DIP/objects_w_caption',
	'''
		
	return config
