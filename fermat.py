import matplotlib.pyplot as plt

started = False

x = []
y = []
choice = 0
trivialCount=0
nonTrivialCount=0
    
def fermat(c,maxval):
    global trivialCount
    global nonTrivialCount

    for a in range(0,int(maxval)+1):
        for b in range(0,int(maxval)+1):
            f = a*a+b*b
            if(f ==  int(c)*int(c) and a>=b and choice==0):

                if(a==0 or b==0):
                    print('a='+str(a)+'\nb='+str(b)+'\tTRIVIAL\n')
                    print(str(a)+'² + '+str(b)+'² = ' + str(c) +'²\n'+20*'-')
                    trivialCount+=1

                else:
                    print('a='+str(a)+'\nb='+str(b)+'\n')
                    print(str(a)+'² + '+str(b)+'² = ' + str(c) +'²\n'+20*'-')
                    nonTrivialCount+=1
                    
            if((a!=0 and b!=0) and f ==  int(c)*int(c) and a>=b and choice==1):
                print('a='+str(a)+'\nb='+str(b)+'\n')
                print(str(a)+'² + '+str(b)+'² = ' + str(c) +'²\n'+20*'-')
                nonTrivialCount+=1
            if(f ==  int(c)*int(c) and choice==2):
                y.append(b)
                x.append(a)
            if(f ==  int(c)*int(c) and a>=b and choice==3):
                if(a==0 or b==0):
                    print('a='+str(a)+'\nb='+str(b)+'\tTRIVIAL\n')
                    print(str(a)+'² + '+str(b)+'² = ' + str(c) +'²\n'+20*'-')
                    trivialCount+=1
                else:
                    print('a='+str(a)+'\nb='+str(b)+'\n')
                    print(str(a)+'² + '+str(b)+'² = ' + str(c) +'²\n'+20*'-')
                    nonTrivialCount+=1

def specific():
    c = input('a²+b² = c²\n\tc = ')
    fermat(c,c)

def begin():
    print('                                                                                                                                                                                    \n                                 :::::.                               .:::                       ,::::,                                                          .::::,             \n                               ;!+;;;;*?,                             ,+S@,                    :!+;;;;*?:                                                      ,!+;;;;+?+           \n                               ..:;;;;*?,               :+              ?@,                     .:;;;;+?:                                                       .:;;;;+?+           \n        ,:!!!!!!!!*,.          *S!****;,                %@.             ?@;:!!!!!!!:,          ;S!****+,.        .,,,,,,,,,,,,,,,.        .:*!!!!!!::%!        :S?****+,.           \n        ;+  ......:@?          ;*++++++*.         ......%@,......       ?@!+.     .+**;        :**+++++*,        ;***************,      :***.      ;*@S        ,**+++++*:           \n        ,:;+!!!!!!?@?                            +!!!!!!#@?!!!!!!.      ?@.         ,@?                                                 *@:          ..                             \n      +?+;;:      .@?                                   %@.             ?@.         ,@?                          ;!!!!!!!!!!!!!!!,      *@:                                         \n      ?@,         ,@?                                   %@.             ?@*+       +*!;                          ................       :!*+.        +;                             \n      .,*!!!!!!!!!*,;!:                                 ,,            :!%S;:!!!!!!!:,                                                     ,:!!!!!!!!!:,                             \n                                                                                                                                                                                    \n                                                                                                                                                                                    \n                                                              ')
    print('This is the Fermat Diophantine rootfinder software')
    print('Written by Brendon Maia Lopes - brendonmaial@poli.ufrj.br')
    print('\n'+20*'-'+'\na²+b²=c²')

def menu():
    print(20*'-'+'\n0 - Find all roots \n1 - Find all non-Trivial roots \n2 - Plot roots \n3 - Find roots for specific \'c\' value\n')
    global choice
    choice = int(input(20*'-'+'\nChoose option:'))
    
def run():
    if choice!=3:
        rangeInput = input('Keep iterating until:')
        for i in range(int(rangeInput)+1):
            fermat(i,i)
            if choice==2:
                print('Calculating '+str(i)+'/'+str(rangeInput)+'...')
        if choice==2:
            plt.scatter(x,y,color='red')
            plt.show()
    elif choice==3:
        specific()

    global trivialCount
    global nonTrivialCount

    if(choice!=2):
        print('Number of roots calculated:\n\tTrivial:'+str(trivialCount)+'\n\tNon trivial:'+str(nonTrivialCount)+'\n\tTotal:'+str(trivialCount+nonTrivialCount))

def main():
    global started
    if(not started):
        begin()
    started = True
    menu()
    run()
    
while __name__ == '__main__':
    try:
        main()
    except:
        main()
        
