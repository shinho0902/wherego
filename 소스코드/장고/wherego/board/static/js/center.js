
$(document).ready(function (){

    var testForm = $("#testForm");
    var index = 1;

    $("#insertButton").on("click", function (){
        if(index == 6){
            alert("최대 5개까지 입력 가능합니다.");
            return false;
        };

        var newDiv = document.createElement("div");
        newDiv.setAttribute("class", "newDiv");

        var newInput = document.createElement("input");
        newInput.setAttribute('size',20);
        newInput.setAttribute("id", "address_kakao"+index);//인덱스 추가하고 싶으면 +index
        newInput.setAttribute("type", "text");
        newInput.setAttribute("name", "address");





        var removeInput = document.createElement("button");
        removeInput.setAttribute("class", "removeInput");
        removeInput.setAttribute("id", "address_kakao_remove"+index);
        removeInput.textContent = "삭제";

        newDiv.append(newInput);
        newDiv.append(removeInput);
        testForm.append(newDiv);

        index+=1;

    });

    $(document).on("click", ".removeInput", function () {
        $(this).parent(".newDiv").remove();
        resetIndex();
    });

    function resetIndex(){
        index = 1
        testForm.children('div').each(function (){
            var target = $(this).children('input[type=text]')
            target.attr("id", "address_kakao"+index)
            
            index+=1
        });
    };



    $("#insertButton").trigger("click");
    $("#insertButton").trigger("click");

});

