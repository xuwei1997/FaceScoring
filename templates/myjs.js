$(function () {
    $(".file").on('change',function () {
        var formData = new FormData();
        formData.append("file", $(".file")[0].files[0]);
        var file = this.files[0];
        if (window.FileReader) {
            var reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onloadend = function (e) {
                $(".img").attr("src", e.target.result);
            };
        }
        $.ajax({
            url:"/upload",
            type:"POST",
            data:formData,
            processData: false,
            contentType: false,
            success:function (data) {
                document.getElementById('img2').innerHTML=data;
            }
        });
    })
}),
    (function(w){
        function changeBg(params){
            var self = this;
            self.direction = 1;//1为竖平，2为垂直
            self.bodyWidth = self.bodyHeight = self.width = self.height = 1;
            function afterChangeDirection(){
                self.bodyWidth = $(document).width();
                self.bodyHeight = $(document).height();
                if(self.direction == 1){
                    document.body.style="";
                }else{
                    var h = self.bodyWidth / (self.width/self.height);
                    h = Math.max(self.height,h);
                    $("body").height(h + "px");
                    document.body.style.backgroundSize=self.bodyWidth+"px "+h + "px";
                }
            }
            function setDirection(dir){
                self.direction = dir;
                afterChangeDirection();
            }
            function init(){
                $.extend(self,params);
                self.bodyWidth = $(document).width();
                self.bodyHeight = $(document).height();
                if(w.matchMedia) {
                    var mql = w.matchMedia("(orientation: portrait)");
                    if (mql.matches) {// 如果有匹配，则我们处于垂直视角
                        setDirection(1);
                    } else {//水平方向
                        setDirection(2);
                    }
                    mql.addListener(function(m) {
                        if(m.matches) {// 改变到直立方向
                            setDirection(1);
                        } else {// 改变到水平方向
                            setDirection(2);
                        }
                    });
                }else if(typeof(w.orientation) != 'undefined'){
                    w.addEventListener('orientationchange', function(event){
                        if ( w.orientation == 180 || w.orientation==0 ) {
                            setDirection(1);
                        }else {//if( window.orientation == 90 || window.orientation == -90 )
                            setDirection(2);
                        }
                    });
                }
            }
            init();
        }
        w.changeBg = changeBg;
    })(window);


//这个代码放页面里，上面代码放JS文件里
$(function(){
    new changeBg({
        width:1200, //背景图片实际宽度
        height:1000 //背景图片实际高度
    });
});