#!/usr/bin/env python
import DIP_load_config
import os

def main():
	params = DIP_load_config.load_config()
	os.chdir('./captioning_model')
	
	cmd = 'th run_ros.lua '
	#for key in params.keys() :
	#	cmd += '-'+str(key)+' '+str(params[key])
	os.system(	cmd )


if __name__=='__main__':	
	main()
