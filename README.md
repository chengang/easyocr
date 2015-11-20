![recognition](https://raw.githubusercontent.com/chengang/easyocr/master/pics/1.jpg)

![HTTP API](https://raw.githubusercontent.com/chengang/easyocr/master/pics/2.jpg)

Setuping is easy on CentOS7.  

yum install epel-release  
yum install opencv*  
yum install libmicrohttpd*  
yum install libcurl*  
yum install boost  
yum install boost-devel  
yum install clang*  
yum install openssl*  

cd src  
make  
./easyocr 80 /var/www/html/pics/   
