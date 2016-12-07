import os
import math
MARGIN=.0001
def get_relation(point1,point2):
    m=(point2[1]-point1[1])/(point2[0]-point1[0])
    c=point2[1]-m*point2[0]
    return m,c
def get_direction(point1,point2):
    steps=int(abs(math.sqrt((point1[0]-point2[0])**2+abs(point1[1]-point2[1])**2))
              /MARGIN)
    if abs(point1[0]-point2[0]) > abs(point1[1]-point2[1]):

        if point1[0]<point2[0]:
            return "x",MARGIN,steps
        else:
            return "x",-1*MARGIN,steps
    else:

        if point1[1]<point2[1]:
            return "y",MARGIN,steps
        return "y",-1*MARGIN,steps

def download(point,counter,hedding,drive='../G_Images_Train/'):

    print(point)
    link="curl https://maps.googleapis.com/maps/api/streetview?size=600x300\&location="+str(point[0])+","+str(point[1])+"\&heading="+hedding[0]+"\&pitch=-0.76\&key=AIzaSyBYWKtpmiTetu5M8SbPNBARXTpX2vu3vKw >>"+drive+ str(counter)+".1"+".jpg"
    os.system(link)
    link="curl https://maps.googleapis.com/maps/api/streetview?size=600x300\&location="+str(point[0])+","+str(point[1])+"\&heading="+hedding[1]+"\&pitch=-0.76\&key=AIzaSyBYWKtpmiTetu5M8SbPNBARXTpX2vu3vKw >>"+drive+ str(counter)+".2"+".jpg"
    os.system(link)

def downloader(point1,point2,counter,hedding,Drive='../G_Images_Train/'):
    next_point=point1
    m,c=get_relation(point1,point2)
    direction=get_direction(point1,point2)
    print(direction[2])
    for _ in range(direction[2]):
        try:
            if direction[0]=='x':
                next_point[0]=next_point[0]+direction[1]
                next_point[1]=m*next_point[0]+c
            else:
                next_point[1]=next_point[1]+direction[1]
                next_point[0]=(next_point[1]-c)/m
            download(next_point,counter,hedding,Drive)
            counter+=1
        except Exception as ex:
            print("got issue",str(ex))
    return counter


def bunddle_downloadder(coordinates,hedding=["40","280"],Drive='../G_Images_Train/'):

    counter=1400
    #,
    #cordinates=[[43.2162016,-77.7415286],[43.2165061,-77.7442845],[43.2163997,-77.7433639],[43.217061,-77.7493902], [43.217431,-77.7525678]]
    #cordinates=[[43.1334352,-78.193272],[43.1484239,-78.1935239]]
    for section in range(len(coordinates)-1):
        start_coordinate=coordinates[section]
        end_coordinate=coordinates[section+1]
        #parameters=get_relation(start_coordinate,end_coordinate)
        counter=downloader(start_coordinate,end_coordinate,counter+1,hedding,Drive)
        print("done")



