from __future__ import print_function
from calc import calc as real_calc
import sys
import zerorpc

class CalcApi(object):
    def calc(self, text):
        try:
            return real_calc(text)
        except Exception as e:
            return 0.0

    def echo(self, text):
        return text

def parse_port():
    return 4242

def main():
    addr = 'tcp://127.0.0.1:' + str(parse_port())
    s = zerorpc.Server(CalcApi())
    s.bind(addr)
    print('Start running on {}'.format(addr))
    s.run()

if __name__ == '__main__':
    main()