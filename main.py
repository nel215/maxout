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

    def update(self,x,sub):
        abs_sub = abs(sub)
        loss = 0 if abs_sub<self.Eps else abs_sub-self.Eps
        v = numpy.sign(sub)*x
        l = numpy.linalg.norm(v)
        tau = loss/(l*l+0.5/self.P)
        self.weight += tau*v

    def train(self,x,y):
        sub = y-self.predict(x)
        self.update(numpy.append(x,1),sub)

    def predict(self,x):
        return numpy.dot(self.weight,numpy.append(x,1.0))

class MaxOut:
    def __init__(self,Dim,Hid=3):
        self.Dim = Dim
        self.Hid = Hid
        self.pa_regs = [PassiveAggressiveRegression(Dim) for _ in xrange(Hid)]

    def argmax(self,x):
        return
    def train(self,x,y):
        idx = self.argmax(x)
        self.pa_regs[idx].train(x,y)

    def predict(self,x):
        numpy.argmax([self.pa_regs[i].predict(x) for i in xrange(self.Hid)])
        idx = self.argmax(x)
        return idx,self.pa_regs[idx].predict(x)


class StackedMaxOut:
    def __init__(self,Dim,Mid):
        self.Dim = Dim
        self.Mid = Mid
        self.weight = []
        for i in xrange(0,len(self.Dim)-1):
            vis = self.Dim[i]
            hid = self.Dim[i+1]
            mid = self.Mid[i]
            self.weight.append(numpy.random.randn(hid,mid,vis))

    def train(self,x,y):
        alpha = 0.00005
        output = [numpy.append(x,1)]
        argmax = []
        weight = []
        for w in self.weight:
            o = numpy.dot(w,output[-1])
            s = []
            argmax.append(map(numpy.argmax,o))
            for i,a in enumerate(argmax[-1]):
                s.append(w[i][a])
            output.append(map(numpy.max,o))
            weight.append(s)



        delta = [(y-output[-1])]
        for i in xrange(len(self.weight)-1,0,-1):
            #print delta[-1],weight[i]
            d = numpy.dot(delta[-1],weight[i])
            delta.append(d)

        delta.reverse()
        for i in xrange(len(self.weight)):
            weight[i] += alpha*numpy.dot(numpy.transpose([delta[i]]),[output[i]])

            for j,a in enumerate(argmax[i]):
                #print self.weight[i][j][a],weight[i][j]
                self.weight[i][j][a] = weight[i][j]

    def predict(self,x):
        output = [numpy.append(x,1)]
        for w in self.weight:
            o = numpy.dot(w,output[-1])
            output.append(numpy.max(o))
        return output[-1]


if __name__=='__main__':
    data = get_data('winequality-red.csv')
    Dim = len(data[0][0])
    pa_reg = PassiveAggressiveRegression(Dim)
    for iter in xrange(10):
        error = 0.0
        for x,y in data:
            error += abs(y-pa_reg.predict(x))
        print error
        for x,y in data:
            pa_reg.train(x,y)
    #
    # maxout = MaxOut(Dim)
    # for iter in xrange(10):
    #     error = 0.0
    #     for x,y in data:
    #         error += abs(y-maxout.predict(x))
    #     print error
    #     for x,y in data:
    #         maxout.train(x,y)

    smo = StackedMaxOut([Dim+1,1],[2,3])
    for iter in xrange(4000):
        error = 0.0
        if iter%20==0:
            for x,y in data:
                error += abs(y-smo.predict(x))
            print error
        for x,y in data:
            smo.train(x,numpy.array([y]))
