from __future__ import print_function

import usb.core
import usb.util
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

vid = 0x1234
pid = 0xed02

dev = usb.core.find(idVendor=vid, idProduct=pid)
if dev is None:
    raise ValueError('Device not found')

if dev.is_kernel_driver_active(1):
    dev.detach_kernel_driver(1)

usb.util.claim_interface(dev, 1)
print(dev.get_active_configuration())

# get an endpoint instance
cfg = dev.get_active_configuration()
intf = cfg[(1, 0)]

ep = usb.util.find_descriptor(intf, custom_match=lambda e: usb.util.endpoint_direction(
    e.bEndpointAddress) == usb.util.ENDPOINT_IN)
assert ep is not None
print("ep:" + str(ep))

print(f"bEndpointAddress: {hex(ep.bEndpointAddress)}\nwMaxPacketSize: {ep.wMaxPacketSize}")

serial_number = str(usb.util.get_string(dev, dev.iSerialNumber))
print(f"Serial Number: {serial_number}")
sn = bytearray([ord(serial_number[i]) for i in range(0, len(serial_number))])

if len(sn) != 16:
    raise Exception('sn has wrong length')

k = ['\0'] * 16

k = [sn[-2], sn[-1], sn[-2], sn[-1], sn[-3], sn[-4], sn[-3], sn[-4],
     sn[-4], sn[-3], sn[-4], sn[-3], sn[-1], sn[-2], sn[-1], sn[-2]]

print(k, bytearray(bytearray(k)))
samplingRate = 256
channels = 40

cipher = AES.new(bytearray(bytearray(k)), AES.MODE_ECB)

# v
# XXXX XXXX | XXXX XXX0
# x[z] x[z+1]
decode_my_shit = lambda x, z: "%.8f" % (
        ((int(str(x[z])) * .128205128205129) + 4201.02564096001) + ((int(str(x[z + 1])) - 128) * 32.82051289))

while True:
    try:
        data = dev.read(ep.bEndpointAddress, ep.wMaxPacketSize)
        decoded_data = cipher.decrypt(data.tobytes())
        counter = [decoded_data[0], decoded_data[1]]
        packet_data = []
        # datamode is never assigned this way in cykit...
        if counter[1] == '16':
            datamode = 1
        elif counter[1] == '32':
            datamode = 2
        else:
            datamode = 0

        for j in range(2, 16, 2):
            edk_value = str(decode_my_shit(decoded_data, j))
            packet_data.append(edk_value)

            # skip decoded_data[16], decoded[17]

        for j in range(18, len(decoded_data), 2):
            edk_value = str(decode_my_shit(decoded_data, j))
            packet_data.append(edk_value)
            # add "contact quality" values to the packet_data array
            # grab UNDECODED decoded_data[16], decoded[17]
            packet_data.append(str(decoded_data[16]))
            packet_data.append(str(decoded_data[17]))

            # if datamode == "16" or datamode == "32":
        print(f'Counter: {counter}')
        print(f'Data Mode: {datamode}')
        print('Data:')
        print(','.join(packet_data))
        print("-" * 200)  # pylint: fuck this magic number
    except usb.core.USBError as e:
        if e.errno == 10060:
            print('timed out...')

# print("Opening the device")

# h = hid.device(0x1234, 0xed02)
# h.open(0x1234, 0xed02)

# print(h)

# print("Manufacturer: %s" % h.get_manufacturer_string())
# print("Product: %s" % h.get_product_string())
# print("Serial No: %s" % h.get_serial_number_string())

# # enable non-blocking mode
# h.set_nonblocking(1)

# print(h.get_feature_report(32, 32))

# try:
# while True:
# print(h.get_feature_report(32, 32))
# d = h.read(64)
# print('read: "{}"'.format(d))
# time.sleep(1)
# finally:
# print("Closing the device")
# h.close()

# h.ctrl_transfer(0xA1, 0x01, 0x0100, 0, 32)
