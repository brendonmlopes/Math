import math

class Vector:
    def __init__(self,x,y,z=0):
        self.x = x
        self.y = y
        self.z = z
        self.mag = (x**2+y**2+z**2)**(1/2)
        
    def __add__(self,w):
        newX = self.x+w.x
        newY = self.y+w.y
        newZ = self.z+w.z
        result = Vector(newX,newY,newZ)
        return result

    def __sub__(self,w):
        w *= (-1)
        result = self+w
        return result

    def __mul__(self,n):
        newX = self.x*n
        newY = self.y*n
        newZ = self.z*n
        result = Vector(newX,newY,newZ)
        return result

    def add(self,w):
        newX = self.x+w.x
        newY = self.y+w.y
        newZ = self.z+w.z
        result = Vector(newX,newY,newZ)
        return result

    def sub(self,w):
        w *= (-1)
        result = self.add(w)
        return result

    def dot(self,w):
        prod = self.x*w.x+self.y*w.y+self.z*w.z
        return prod

    def cross(self,w):
        newX = self.y*w.z-self.z*w.y
        newY = self.z*w.x-self.x*w.z
        newZ = self.x*w.y-self.y*w.x
        prod = Vector(newX,newY,newZ)
        return prod

    def show(self):
        print('[',self.x,']')
        print('[',self.y,']')
        print('[',self.z,']')
        print('')
        return None
    
    def rotate2D(self,angle):
        R = Matrix([[math.cos(angle),math.sin(angle)],[-math.sin(angle),math.cos(angle)]])
        V = Matrix([[self.x],[self.y]])
        P = R*V
        self.x = P.elements[0][0]
        self.y = P.elements[1][0]
        return None

    def rotateW(self,w,angle):
        '''Rotates the vector around the vector w'''
        a=angle
        c=math.cos(a)
        s=math.sin(a)
        t=1-math.cos(a)
        x=w.x/w.mag
        y=w.y/w.mag
        z=w.z/w.mag    

        R = Matrix([[t*x**2+c , t*x*y-s*z , t*x*z+s*y , 0],
                    [t*x*y+s*z, t*y**2+c  , t*y*z-s*x , 0],
                    [t*x*z-s*y, t*y*z+s*x , t*z**2+c  , 0],
                    [    0    ,     0     ,     0     , 1]])
        V =  Matrix([[self.x],[self.y],[self.z],[0]])
        P = R*V
        
        self.x = P.elements[0][0]
        self.y = P.elements[1][0]
        self.z = P.elements[2][0]
        return None
    
    def getAngles(self):
        theta = math.atan2(self.y,self.x)
        phi = math.atan2(self.z,(self.y**2+self.x**2)**(1/2))
        return (theta, phi)
    
    def getCoord(self):
        vCoord = (self.x,self.y,self.z)
        return vCoord
        
class Matrix:
    def __init__(self,elementsList):
        self.elements=elementsList
        self.nRows=len(self.elements)
        self.nCols=len(self.elements[0])
        
    def __mul__(self,A):
        if(self.nCols!=A.nRows):
            raise Exception('The number of rows of the first matrix must match the number of rows of the second matrix')

        result=[]
        for i in range(self.nRows):
            result.append([])
            for j in range(A.nCols):
                result[i].append(0)
        
        for i in range(self.nRows):
            for j in range(A.nCols):
                for k in range(A.nRows):
                    result[i][j] += self.elements[i][k] * A.elements[k][j]
                    
        result=Matrix(result)
        return result

    
    def show(self):
        for i in self.elements:
            print(i)
        return None

