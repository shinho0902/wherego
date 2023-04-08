from django.shortcuts import render, redirect
from account.models import User
from .models import Post

def login_required(function):
    # wrapping한 함수와 기존 함수의 인자값을 맞춰 줘야 한다
    def wrap(request, *args, **kwargs):
        user = request.session.get('user')
        if user is None or not user:
            return redirect('account:login')
        return function(request, *args, **kwargs)

    return wrap

def auth_required(function):
    # wrapping한 함수와 기존 함수의 인자값을 맞춰 줘야 한다
    def wrap(request, *args, **kwargs):
        user = request.session.get('user')
        if user is None or not user:
            return redirect('account:login')
        
        try:
            post_user = User.objects.get(username=user)
            post = Post.objects.get(id=kwargs["id"])
            if post_user.username != post.writer.username:
                return redirect('board:index', page=1)
        except:
            return render(request, 'board/error.html')
        return function(request, *args, **kwargs)

    return wrap

