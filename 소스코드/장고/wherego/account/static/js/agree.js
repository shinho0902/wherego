$("#chk-all").click( function() {
    var checkedVal = $(".chk").prop("checked");

    $(".chk").prop("checked", this.checked);
    if($("#chk-all").prop("checked") == true) {
      $(".check-msg").css("color", "#43494f");
    } else {
      $(".check-msg").css("color", "#b0b5c1");
    }
  });

  $(".chk").on("click", function() {
    var checked = $(this).attr('id');
    var checkedColor = $(`.${checked}`).css("color");

    if(checkedColor == 'rgb(176, 181, 193)') {
      $(`.${checked}`).css("color", "#43494f");
    } else {
      $(`.${checked}`).css("color", "#b0b5c1");
    }

  });