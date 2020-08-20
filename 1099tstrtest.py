import RPi.GPIO as GPIO
import threading
import subprocess
import os
import time
import signal
import atexit
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


global p
# global depthaiThread

def cleanup():
    # global depthaiThread
    if(p is not None):
        print('Stopping subprocess with pid: ', str(p.pid))
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        print('Stopped!')

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
    print("Applying power to test board")
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
        return(False)
    else:
        return(True)

def rundepthai():
    # args=""
    # for arg in sys.argv[1:]:
    #     args+="'"+arg+"' "
    global p
    # test_cmd = """python3 test.py -s left,10 right,10 previewout,10"""
    test_cmd = """python3 depthai.py -s left,10 right,10 previewout,10"""
    p = subprocess.Popen(test_cmd, shell=True, preexec_fn=os.setsid)
    return_code = p.returncode
    print("Return code:"+str(return_code))
    

# def main():
#     global p
#     global depthaiThread
#     depthaiThread = threading.Thread(target=rundepthai) # setting rundepthai to run on therad
#     # isDetected = False
#     while True:
#         if moddetect() and not depthaiThread.is_alive(): # Starting child process in a thread if device detected and thread is not alive
#             isDetected = True
#             print("Module detected!")
#             forcepoweron()
#             rst()
#             print("Starting test run...")
#             depthaiThread.start()
#         elif not moddetect() and depthaiThread.is_alive(): # stoping child process if device is disconnected and thread is alive
#             isDetected = False
#             print("Module unplugged!!!")
#             print("Killing test run...")
#             p.kill()
#             # if depthaiThread.is_alive()
#             depthaiThread.join()
#         elif not moddetect() and not depthaiThread.is_alive():
#             isDetected = False
#             print("No module detected. Please mount BW1099 module and retry.")
        
# def main():
#     global p
#     isDetected = False
#     i = 0;
#     rundepthai()
#     time.sleep(10)
#     while True:
#         if isDetected: # stoping child process if device is disconnected and thread is alive
#             isDetected = False
#             print("Module unplugged!!!")
#             print("Killing test run...")
#             p.kill()

#         if(i == 10000):
#             isDetected = True
#         i += 1

def main():
    global p
    isDetected = False
    while True:
        if moddetect() and not isDetected: # Starting child process if device detected and child process is not started.
            isDetected = True
            print("Module detected!")
            forcepoweron()
            rst()
            print("Starting test run...")
            rundepthai()
        elif not moddetect() and isDetected: # stoping child process if device is disconnected and child process is alive.
            isDetected = False
            print("Module unplugged!!!")
            print("Killing test run...")
            p.kill()
            p = None

        elif not moddetect() and not isDetected:
            print("No module detected. Please mount BW1099 module and retry.")

atexit.register(cleanup)

if __name__== "__main__":
  main()


