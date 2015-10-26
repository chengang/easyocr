all:
	clang++ main.cpp httpcurl.cpp easyocr.cpp webserver.cpp webcontroller.cpp helper.cpp -lboost_system -lboost_filesystem `pkg-config --cflags --libs opencv openssl libmicrohttpd libcurl`
run:
	./a.out http://static.xiaojukeji.com/gulfstream/upload/2015/20150305/car_72458/licensefimg.jpg
clean:
	rm -f ./a.out

