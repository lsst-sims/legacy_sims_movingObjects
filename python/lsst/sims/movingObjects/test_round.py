import numpy as np


class TestRounding():

    def __init__(self, tStart, tSpan, nDecimal=10):
        self.nDecimal = nDecimal
        self.tStart = round(tStart, self.nDecimal)
        self.tSpan = round(tSpan, self.nDecimal)

    def _roundLength(self, length):
        """Modify length, to fit in an 'integer multiple' within the tStart/tEnd,
        and to have the desired number of decimal values.

        Parameters
        ----------
        length : float
            The input length value to be rounded.

        Returns
        -------
        float
            The rounded length value.
        """
        length_in = length
        length = round(length, self.nDecimal)
        # Make length an integer value within the time interval.
        tolerance = 1.0e-30
        if self.nDecimal is not None:
            tolerance = 10.**(-1*self.nDecimal)
        counter = 0
        prev_int_factor = 0
        while ((self.tSpan % length) > tolerance) and (length > 0) and (counter < 10):
            int_factor = int(self.tSpan / length) + 1
            if int_factor == prev_int_factor:
                int_factor = prev_int_factor + 1
            print counter, int_factor, prev_int_factor, self.tSpan%length, self.tSpan/length, length, tolerance
            prev_int_factor = int_factor
            length = self.tSpan / int_factor
            counter += 1
        if (self.tSpan % length) > tolerance:
            # Add this entire segment into the failed list.
            #for objId in self.orbitsObj.orbits['objId'].as_matrix():
            #    self.failed.append((objId, self.tStart, self.tEnd))
            raise ValueError('Could not find a suitable length for the timespan (start %f, span %f), starting with %f'
                             % (self.tStart, self.tSpan, length_in))
        return length



if __name__ == '__main__':

    tStart = 50099.831278
    tEnd = 50100.575172
    tspan = tEnd - tStart

    testR = TestRounding(tStart, tspan, nDecimal=10)

    lengths = np.arange(0.1, (tEnd-tStart), 0.0001)
    for length in lengths:
        testR._roundLength(length)
        print 'successful at %f' % length

    tStart = 50000.00
    tEnd = 50060.00
    tspan = tEnd - tStart

    testR = TestRounding(tStart, tspan, nDecimal=10)

    lengths = np.arange(0.1, 1.0, 0.0001)
    for length in [0.719483]:
        testR._roundLength(length)
        print 'successful at %f' % length


