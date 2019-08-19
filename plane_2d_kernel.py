import matplotlib.pyplot as plt
import numpy as np

width   = 3000.
spacing = 10.

velo    = 4.
T       = np.array([80.])
amp     = np.array([1.])
freq    = 1./T

# setup the grid
x   = np.arange(width/spacing+1)*spacing - width/2.
N = x.size
y   = x.copy()
x, y= np.meshgrid(x,y)
# x   = 
# x0=x;
# n=numel(x);
# x=x(ones(n,1),:);
# y=x';
# 
# % normalize amps
# a=a/sum(a);


k   = 2.*np.pi*6371.*freq/velo;
r   = np.sqrt(x**2+y**2)

r[r==0.] = np.sqrt(2.)*spacing
# # r(floor(n*n/2)+1)=sqrt(2)*d; # avoid divide by zero at receiver

c   = spacing **2/(4*6371.**2 *np.sqrt(2*np.pi*abs(np.sin(r/6371.))));
Kph = np.zeros(N);
Kam = np.zeros(N);

for i in range(k.size):
    Kph = Kph-c*amp[i]*k[i]**1.5 *np.sin( k[i]*(x+r)/6371. + np.pi/4. )
    
x   = x - 100.
r   = np.sqrt(x**2+y**2)
r[r==0.] = np.sqrt(2.)*spacing
c   = spacing **2/(4*6371.**2 *np.sqrt(2*np.pi*abs(np.sin(r/6371.))));
Kph2 = np.zeros(N);
for i in range(k.size):
    Kph2 = Kph2-c*amp[i]*k[i]**1.5 *np.sin(k[i]*(x+r)/6371.+np.pi/4.);

# for i=1:numel(f)
#     Kph=Kph-c.*a(i).*k(i).^1.5.*sin(k(i).*(x+r)./6371+pi/4);
#     Kam=Kam-c.*a(i).*k(i).^1.5.*cos(k(i).*(x+r)./6371+pi/4);
# end