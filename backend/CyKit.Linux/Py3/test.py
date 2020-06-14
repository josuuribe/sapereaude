import usb.core
import usb.util
import os
import sys
import usb.util
import time


def active():
    if dev.is_kernel_driver_active(0):
        print("Interface 0 active")
    else:
        print("Interface 0 inactive")

    if dev.is_kernel_driver_active(1):
        print("Interface 1 active")
    else:
        print("Interface 1 inactive")


def attach(device):
    if not dev.is_kernel_driver_active(0):
        print("detach 0")

        dev.attach_kernel_driver(0)
    # if not dev.is_kernel_driver_active(1):


#	dev.attach_kernel_driver(1)


def detach(device):
    if dev.is_kernel_driver_active(0):
        dev.detach_kernel_driver(0)


# if dev.is_kernel_driver_active(1):
#	dev.detach_kernel_driver(1)

os.environ['PYUSB_DEBUG'] = "info"
os.environ['PYUSB_LOG_FILENAME'] = "/home/pi/cykit/CyKit/Py3/log.log"

# find our device
dev = usb.core.find(idVendor=0x1234, idProduct=0xed02)

# was it found?
if dev is None:
    raise ValueError('Device not found')

print(str(dev))

print("======================================")

active()
# usb.util.claim_interface(dev, 1)
# detach(dev)
# dev.reset()
# dev.set_configuration()
result = usb.control.get_status(dev)
print("Result:" + str(result))

# interface = usb.control.get_interface(dev, 1)
# print("Interface:" + str(interface))

while 1:
    detail_info = list(dev.ctrl_transfer(0xA1, 0x01, 0x0300, 0, 32))
    print(str(detail_info))
    data = ""
    data = dev.read(0x82, 32, 1000)
    if data != "":
        print(">>>" + str(list(data)))
# time.sleep(.1)
print("ctrl" + str(detail_info))
task = dev.read(0x82, 32, 500)
print(str(task))
exit(0)

exit(0)
cfg = dev.get_active_configuration()
print("cfg:" + str(cfg))

bmRequestType = usb.util.build_request_type(
    usb.util.CTRL_IN,
    usb.util.CTRL_TYPE_STANDARD,
    usb.util.CTRL_RECIPIENT_DEVICE)
print("request:" + str(bmRequestType))
dev.ctrl_transfer(
    bmRequestType=bmRequestType,
    bRequest=0xA1
)
exit(0)

alt = usb.util.find_descriptor(cfg, find_all=True)

for a in alt:
    print("a:" + str(a))

# for a in cfg:
#    print(str(a))

"""
for cfg in dev:
    sys.stdout.write(str(cfg.bConfigurationValue) + '\n')
    for intf in cfg:
        sys.stdout.write('I.\t' + \
                         str(intf.bInterfaceNumber) + \
                         'A.,' + \
                         str(intf.bAlternateSetting) + \
                         '\n')
        for ep in intf:
            sys.stdout.write('E.\t\t' + \
                             str(ep.bEndpointAddress) + \
                             '\n')
"""

"""
dev.attach_kernel_driver(0)
dev.attach_kernel_driver(1)
exit(0)
"""

# detach(dev)
# dev.reset()

# set the active configuration. With no arguments, the first
# configuration will be the active one

# dev.set_configuration()
print("configuration set")
# cfg = usb.util.find_descriptor(dev, bConfigurationValue=1)


# get an endpoint instance
cfg = dev.get_active_configuration()
intf = cfg[(1, 0)]

ep = usb.util.find_descriptor(
    intf,
    # match the first OUT endpoint
    custom_match= \
        lambda e: \
            usb.util.endpoint_direction(e.bEndpointAddress) == \
            usb.util.ENDPOINT_IN)

print("ep:", ep)

# detach(dev)
# detach(dev)
# dev.set_interface_altsetting(interface = 1, alternate_setting = 0)
# attach(dev)

x = dev.reset()
# dev.set_interface_altsetting(interface = 1, alternate_setting = 0)
# detail_info = list(dev.ctrl_transfer(0xA1, 0x01, 0x0300, 1, 31))
detail_info = dev.ctrl_transfer(0xC0, CTRL_LOOPBACK_READ, 0, 0, 100)
print("control set:" + str(detail_info))

# attach(dev)


# print("reset:" + str(x))
dev.write(0x81, 'test')
# attach(dev)
print("before read:")
task = dev.read(0x82, 32, 500)
# detach(dev)
while 1:
    try:
        task = dev.read(0x82, 32, 500)
        print(str(task))
    except:
        continue
# attach(dev)
print("task:" + task)

print(str(ep))
assert ep is not None
