import socket
import selectors
import types
from kernel.Interfaces.IProducer import IProducer
from kernel.Warden import Warden
import managers
import numpy as np
from multiprocessing import Value
from ctypes import c_bool
import datetime


class TcpProducer(IProducer):
    """
    This class is a TCP Producer, connects to Cykit and gets information to be processed.
    """

    def __init__(self, warden: Warden):
        """
        Constructor that initializes entity
        :param warden: Warden that manages everything.
        """
        self.host = managers.get_host()
        self.port = managers.get_port()
        self.buffer_size = managers.get_buffer_size()
        self.messages = [b"start"]
        self.rest = ""
        self.sel = selectors.DefaultSelector()
        self.warden = warden
        self.text_buffer = b''
        self.__run__ = True
        self.__start__ = Value(c_bool, self.__run__)

    def start_connections(self, host, port):
        """
        Start connection to Cykit.
        :param host: Host to connect to.
        :param port: Port used to connect.
        """
        server_addr = (host, port)
        print("Iniciando conexión id = ", 0, " a ", server_addr)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        sock.connect_ex(server_addr)
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        data = types.SimpleNamespace(
            connid=0,
            msg_total=sum(len(m) for m in self.messages),
            recv_total=0,
            messages=list(self.messages),
            outb=b"",
        )
        self.sel.register(sock, events, data=data)

    def buffer(self, message):
        """
        Process all incoming messages:
        - Splits them in lines
        - Converts it in a numpy array
        - Removes extra column
        - Save extra information to append to next message
        - Put information in queue
        :param message: Data to process
        """
        try:
            decoded_message = message.decode()
            decoded_message = self.rest + decoded_message
            lines = decoded_message.splitlines()
            if not decoded_message.endswith(("\r", "\n", "\r\n")):
                self.rest = lines.pop(-1)

            data = list()

            for line in lines:
                splitted = line.split(",")
                if len(splitted) == 16:
                    data.append(splitted)

            if len(data) > 0:
                x = np.array(data)
                y = x.astype(np.float)
                z = np.delete(y, [1], axis=1)

                for line in z:
                    self.warden.__queue__.put(line)
                    if len(line) < 15:
                        print("Línea incorrecta")
                self.text_buffer = b''
            self.warden.lock()
        except Exception as e:
            print('Error del cliente TCP:', e)

    def service_connection(self, key, mask):
        """
        Manages connection.
        :param key: Socket
        :param mask: Connection status
        :return:
        """
        sock = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(self.buffer_size)  # Should be ready to cls read
            if recv_data:
                self.buffer(recv_data)
                data.recv_total += len(recv_data)
            if not recv_data or data.recv_total == data.msg_total:
                print("Cerrando conexión id = ", data.connid)
                self.sel.unregister(sock)
                sock.close()
        if mask & selectors.EVENT_WRITE:
            if not data.outb and data.messages:
                data.outb = data.messages.pop(0)
            if data.outb:
                sent = sock.send(data.outb)  # Should be ready to write
                data.outb = data.outb[sent:]

    def start(self):
        """
        Start TCP connection
        """
        print("Arrancando productor TCP -> {}.".format(datetime.datetime.now().time()))
        self.start_connections(self.host, int(self.port))
        self.__start__.value = True
        self.warden.__event__.clear()
        try:
            while self.__start__.value:
                events = self.sel.select(timeout=1)
                if events:
                    for key, mask in events:
                        self.service_connection(key, mask)
                if not self.sel.get_map():  # Check for a socket being monitored to continue.
                    break
            print("Productor TCP cerrado -> {}.".format(datetime.datetime.now().time()))
        except KeyboardInterrupt:
            print("Interrupción por teclado.")
        except Exception as e:
            print("Error en TCP: ", e)
        finally:
            self.sel.close()

    def stop(self):
        """
        Stop TCP Producer
        :return:
        """
        self.__start__.value = False
        self.warden.__event__.set()
