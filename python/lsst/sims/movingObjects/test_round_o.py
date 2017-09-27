import numpy as np

class TestRounding():

    def __init__(self, tStart, tEnd, nDecimal=14):
        self.nDecimal = nDecimal
        self.tStart = round(tStart, self.nDecimal)
        self.tEnd = round(tEnd, self.nDecimal)

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
        # Make length an integer value within the time interval.
        timespan = self.tEnd - self.tStart
        tolerance = 1.0e-30
        if self.nDecimal is not None:
            length = round(length, self.nDecimal)
            tolerance = 10.**(-1*self.nDecimal)
        counter = 0
        prev_int_factor = 0
        while ((timespan % length) > tolerance) and (length > 0) and (counter < 10):
            int_factor = np.ceil(timespan / length)
            if int_factor == prev_int_factor:
                int_factor = prev_int_factor + 1
            #print counter, int_factor, prev_int_factor, timespan%length, timespan/length, length, tolerance
            prev_int_factor = int_factor
            length = timespan / int_factor
            if self.nDecimal is not None:
                length = round(length, self.nDecimal)
            counter += 1
        if (timespan % length) > tolerance:
            # Add this entire segment into the failed list.
            #for objId in self.orbitsObj.orbits['objId'].as_matrix():
            #    self.failed.append((objId, self.tStart, self.tEnd))
            raise ValueError('Could not find a suitable length for the timespan (%f to %f: D %f), starting with %f'
                             % (self.tStart, self.tEnd, self.tEnd-self.tStart, length_in))
        return length



if __name__ == '__main__':

    tStart = 50099.831278
    tEnd = 50100.575172
    tspan = tEnd - tStart
    tStart = round(tspan, 14)
    tEnd = tStart + round(tspan, 14)

    testR = TestRounding(tStart, tEnd, nDecimal=14)

    lengths = np.arange(0.2, (tEnd-tStart), 0.0001)
    for length in lengths:
        testR._roundLength(length)
        print 'successful at %f' % length


