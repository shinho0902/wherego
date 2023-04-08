function check_pw(){  //비밀번호 확인 
    var p = document.getElementById('pw').value; 
    var p_cf = document.getElementById('pw_cf').value; 
 
    if (p!=p_cf) { 
      document.getElementById('pw_check_msg').innerHTML = "비밀번호가 일치하지 않습니다. 다시 확인해 주세요."; 
      document.getElementById('pw_check_msg').style.color = 'red'
    } 
    else { 
        document.getElementById('pw_check_msg').innerHTML = "비밀번호가 일치합니다."; 
        document.getElementById('pw_check_msg').style.color = 'blue'
    } 
    if (p_cf=="") { 
      document.getElementById('pw_check_msg').innerHTML = ""; 
    }
 } 

function pw_cond(){ 
  var p = document.getElementById('pw').value; 
  const regExp = /^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$/dim;
  if (p.length < 10 || p.length >20 ){
    document.getElementById('pw_pro_label').innerHTML = "비밀번호는 10~20자리 이내로 입력해주세요.";
    document.getElementById('pw_pro_label').style.color = 'red'
  }
  if (p.length > 9 && p.length < 21 ){
    if (regExp.test(p)){
      document.getElementById('pw_pro_label').innerHTML = "사용 가능한 비밀번호입니다.";
      document.getElementById('pw_pro_label').style.color = 'blue'
    } else{
      document.getElementById('pw_pro_label').innerHTML = "숫자, 문자, 특수문자를 1개 이상씩 포함시켜주세요.";
      document.getElementById('pw_pro_label').style.color = 'red'
    }
  }
 }

 function checkValidDate() {
	var result = true;
	try {
	    var y = parseInt(document.getElementById('year').value),
	        m = parseInt(document.getElementById('month').value),
	        d = parseInt(document.getElementById('day').value);
	    
	    var dateRegex = /^(?=\d)(?:(?:31(?!.(?:0?[2469]|11))|(?:30|29)(?!.0?2)|29(?=.0?2.(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00)))(?:\x20|$))|(?:2[0-8]|1\d|0?[1-9]))([-.\/])(?:1[012]|0?[1-9])\1(?:1[6-9]|[2-9]\d)?\d\d(?:(?=\x20\d)\x20|$))?(((0?[1-9]|1[012])(:[0-5]\d){0,2}(\x20[AP]M))|([01]\d|2[0-3])(:[0-5]\d){1,2})?$/;
	    result = dateRegex.test(d+'-'+m+'-'+y);
	} catch (err) {
      document.getElementById('birth_check_msg').innerHTML = "생년월일을 확인해주세요."; 
      document.getElementById('birth_check_msg').style.color = 'red'
	}    
  if (result){
      document.getElementById('birth_check_msg').innerHTML = ""; 
  }else{
      document.getElementById('birth_check_msg').innerHTML = "생년월일을 확인해주세요."; 
      document.getElementById('birth_check_msg').style.color = 'red'
  }
}



function join_form_check() {
  //변수에 담아주기
  var uid = document.getElementById("id_pro_label").innerText;
  var pwd = document.getElementById("pw_pro_label").innerText;
  var repwd = document.getElementById("pw_check_msg").innerText;
  var uname = document.getElementById("username").value;
  var birth = document.getElementById("birth_check_msg").innerText;
  var gender = $('input:radio[name=gender]').is(':checked');
  var email = document.getElementById("email_pro_label").innerText;

  if (uid != "사용 가능한 닉네임 입니다."){
    alert("닉네임를 확인해주세요.");
  } else if(pwd != "사용 가능한 비밀번호입니다."){
    alert("비밀번호를 확인해주세요.");
  } else if(repwd != "비밀번호가 일치합니다."){
    alert("비밀번호가 일치하지않습니다.");
  } else if(email != "사용 가능한 이메일 입니다."){
    alert("이메일을 다시 입력해주세요.");
  }
  else if(!uname){
    alert("이름을 입력해주세요.");
  } else if(birth){
    alert("생년월일을 확인해주세요.");
  } else if(!gender){
    alert("성별을 선택해주세요.")
  } else{
    document.getElementById('frm').submit();
  }
  }

//  function id_cond(){ 
//   var p = document.getElementById('user').value; 
  
//   if (p.length < 6 || p.length >10 ){
//     document.getElementById('id_pro_label').innerHTML = "아이디는 6~10자리 이내로 입력해주세요.";
//     document.getElementById('id_pro_label').style.color = 'red'
//   }
//   if (p.length > 5 && p.length < 11 ){
//     document.getElementById('id_pro_label').innerHTML = "사용 가능한 아이디 형식입니다.";
//     document.getElementById('id_pro_label').style.color = 'blue'
//   }
//  }
 


//  if(check_SC == 0){
//      document.getElementById('pw_pro_label').innerHTML = '비밀번호에 !,@,#,$,% 의 특수문자를 포함시켜야 합니다.'
//      return;
//  }
//  document.getElementById('pw_pro_label').innerHTML = '';
//  if(pw.length < 8){
//      document.getElementById('pw_pro').value='1';
//  }
//  else if(pw.length<12){
//      document.getElementById('pw_pro').value='2';
//  }
//  else{
//      document.getElementById('pw_pro').value='3';
//  }
//  function check_id(){
//   var id = document.getElementById('user').value;
//  }