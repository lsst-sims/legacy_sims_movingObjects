import decimal
import numpy as np
import time


def dtime(time_prev):
    return (time.time() - time_prev, time.time())

class testTimes():
    def __init__(self):
        nDecimal = 14
        self.tStart = round(52300, nDecimal)
        self.tEnd = round(52330, nDecimal)
        self.length = round(1.58, nDecimal)
        self.timestep = self.length/int(64)

    def getAllTimes(self):
        times = np.arange(float(self.tStart), float(self.tEnd) + float(self.timestep) / 2.0,
                          float(self.timestep), dtype=float)
        return times

class testTimesDecimal():
    def __init__(self):
        nDecimal = 14
        nexp = decimal.Decimal('%s' % round(10 ** -nDecimal, nDecimal))
        self.tStart = decimal.Decimal(52300).quantize(nexp)
        self.tEnd = decimal.Decimal(52330).quantize(nexp)
        self.length = decimal.Decimal(1.58).quantize(nexp)
        self.timestep = self.length / int(64)

    def getAllTimes(self):
        times = np.arange(self.tStart, self.tEnd + self.timestep / 2,
                          self.timestep)
        # dtype=float converts to float, much faster than leaving it off.
        return times


if __name__ == '__main__':

    t = time.time()
    tt = testTimes()
    ntests = 1000
    for i in range(ntests):
        times = tt.getAllTimes()
    dt, t = dtime(t)
    print times.min(), times.max(), 'in seconds:', dt

    t = time.time()
    tD = testTimesDecimal()
    for i in range(ntests):
        times = tD.getAllTimes()
    dt2, t = dtime(t)
    print times.min(), times.max(), 'in seconds:', dt2
    print 'diff', dt2 - dt, (dt2/dt)
