import RPi.GPIO as GPIO
import os
import signal
import subprocess
import time
import itertools
import atexit
import sys
import argparse
from argparse import ArgumentParser


# Set up RPi GPIOs
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)   
GPIO.setup(12,GPIO.OUT)   # TST_PWR_ON
GPIO.setup(13,GPIO.IN)    # MOUNT_SENSE
GPIO.setup(26,GPIO.OUT)   # SYS_RST

# initialize gpios
GPIO.output(12,False)     # power off by default
GPIO.output(26,True)      # not in reset state by default


global all_processes
all_processes=list()
def cleanup():
    global run
    run=False
    timeout_sec = 1
    for p in all_processes: # list of your processes
        print("cleaning")
        p_sec = 0
        for _ in range(timeout_sec):
            if p.poll() == None:
                time.sleep(1)
                p_sec += 1
        if p_sec >= timeout_sec:
            p.killpg()
            # os.killpg(os.getpgid(p.pid), signal.SIGTERM)  # Send the signal to all the process groups
    print('cleaned up!')


def init():
    # Set up RPi GPIOs
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)   
    GPIO.setup(12,GPIO.OUT)   # TST_PWR_ON
    GPIO.setup(13,GPIO.IN)    # MOUNT_SENSE
    GPIO.setup(26,GPIO.OUT)   # SYS_RST

    # initialize gpios
    GPIO.output(12,False)     # power off by default
    GPIO.output(26,True)      # not in reset state by default
    return

def rst(sleeptime=0.1):
    GPIO.output(26,False)     # reset MX just in case
    time.sleep(sleeptime)
    GPIO.output(26,True)
    time.sleep(sleeptime)
    return
    
def pwron():
    #turns on power after checking module is present
    if moddetect()==True:
        print("Applying power to test board")
        GPIO.output(12,True)    # Turn power on to module
        time.sleep(1)    
        return(True)
    else:
        return(False)
    
def forcepoweron():
    #Turns power on without checking module is present
    GPIO.output(12,True)    # Turn power on to module
    time.sleep(1)    
    return   

def pwroff():
    print("Cutting power to test board")
    GPIO.output(12,False)    # Turn power on to module
    time.sleep(1)       
    return

def moddetect():
    if GPIO.input(13)==True: #Mounting module grounds GPIO, so logic is inverted. 
        print("No module detected. Please mount BW1099 module and retry.")
        return(False)
    else:
        print("Module detected!")
        return(True)
        
def rundepthai():
    args=""
    for arg in sys.argv[1:]:
        args+="'"+arg+"' "
    test_cmd = """python3 test.py -s left,10 right,10 previewout,10"""

    atexit.register(cleanup)
    p = subprocess.run(test_cmd, shell=True)
    return_code = p.returncode
    print("Return code:"+str(return_code))
    all_processes.clear()
    

def main():
    while True:
        pwrsuccess = pwron()
        rst()
        if pwrsuccess:
            rundepthai()
            pwroff()
        

if __name__== "__main__":
  main()



    

