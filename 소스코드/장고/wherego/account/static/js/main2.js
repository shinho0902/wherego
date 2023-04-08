

(function($) {

	var	$window = $(window),
		$body = $('body');

	// Breakpoints.
		breakpoints({
			xlarge:  [ '1281px',  '1680px' ],
			large:   [ '981px',   '1280px' ],
			medium:  [ '737px',   '980px'  ],
			small:   [ null,      '736px'  ]
		});

	// Play initial animations on page load.
		$window.on('load', function() {
			window.setTimeout(function() {
				$body.removeClass('is-preload');
			}, 100);
		});

	// Dropdowns.
		$('#nav > ul').dropotron({
			mode: 'fade',
			noOpenerFade: true,
			speed: 300
		});

	// Nav.

		// Toggle.
			$(
				'<div id="navToggle">' +
					'<a href="#navPanel" class="toggle"></a>' +
				'</div>'
			)
				.appendTo($body);

		// Panel.
			$(
				'<div id="navPanel">' +
					'<nav>' +
						$('#nav').navList() +
					'</nav>' +
				'</div>'
			)
				.appendTo($body)
				.panel({
					delay: 500,
					hideOnClick: true,
					hideOnSwipe: true,
					resetScroll: true,
					resetForms: true,
					side: 'left',
					target: $body,
					visibleClass: 'navPanel-visible'
				});

})(jQuery);

let num = 0;
let idx = 0;
let delta;
const mainImg = document.querySelector(".photozone");
const mainTit = document.querySelector(".main_tit01");
const elMainCon = document.querySelectorAll(".main-page");

$(window).on('mousewheel DOMMouseScroll', function (e) {
    delta = e.originalEvent.wheelDelta || e.originalEvent.detail * -1;
    if (delta < 0) {
        if (!(num == 12)) {
            num++;
            if (num <= 11) {
                mainImg.style = `width:${(num * 5) + 50}vw; height:${(num * 5) + 50}vh;`;
            }
        }
        if ((num == 12) && (idx < elMainCon.length)) {
            idx++;
        }
    } else {
        if (!(idx == 0)) {
            idx--;
        }
        if ((num > 0) && (idx == 0)) {
            num--;
            mainImg.style = `width:${(num * 5) + 50}vw; height:${(num * 5) + 50}vh;`;
        }
    }
    
    if (num >= 10) {
        mainTit.style = `color:#fff;`;
        mainImg.classList.add("on");
    } else {
        mainTit.style = `color:#000;`;
        mainImg.classList.remove("on");
    }

    $('html,body').stop().animate({
        scrollTop: ($(window).height()) * idx
    }, 600)
})

