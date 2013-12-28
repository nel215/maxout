#coding:utf-8
import numpy

def get_data(filename):
    f = open(filename)
    # skip title
    f.readline()
    res = []
    for row in f.readlines():
        row = row.split(';')
        y = float(row[-1])
        x = numpy.array(map(float,row[0:-1]))
        res.append((x,y))
    f.close()
    return res

class PassiveAggressiveRegression:
    def __init__(self,Dim,Eps=0.1,P=25.0):
        self.Dim = Dim
        self.Eps = Eps
        self.P = P
        self.weight = numpy.random.randn(self.Dim+1)

    def train(self,x,y):
        x = numpy.append(x,1.0)
        sub = y-numpy.dot(self.weight,x)
        abs_sub = abs(sub)
        loss = 0 if abs_sub<self.Eps else abs_sub-self.Eps
        v = numpy.sign(sub)*x
        l = numpy.linalg.norm(v)
        tau = loss/(l*l+0.5/self.P)
        self.weight = self.weight+tau*v

    def predict(self,x):
        return numpy.dot(self.weight,numpy.append(x,1.0))


if __name__=='__main__':
    data = get_data('winequality-red.csv')
    Dim = len(data[0][0])
    pa_reg = PassiveAggressiveRegression(Dim,Eps=0.05)
    for iter in xrange(200):
        error = 0.0
        for x,y in data:
            error += abs(y-pa_reg.predict(x))
        print error
        for x,y in data:
            pa_reg.train(x,y)
