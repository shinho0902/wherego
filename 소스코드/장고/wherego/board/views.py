from django.http import HttpResponseRedirect
from django.shortcuts import render,redirect
from django.urls import reverse
from django.core.paginator import Paginator
from .decorator import login_required,auth_required
from .models import Post
from account.models import User

@login_required
def index(request,page):
    all_posts = Post.objects.filter(is_active=True).order_by("-id") # 모든 데이터 조회, 내림차순(-표시) 조회
    paginator = Paginator(all_posts, 5)
    board_list = paginator.get_page(page)
    
    return render(request, 'board/index.html', {'board_list':board_list})

@login_required
def detail(request,id,num=1):
    try:
        post = Post.objects.get(id=id)
        if post and post.is_active:
            post.views += 1
            post.save()
            context = {}
            if post.writer.username == User.objects.get(username=request.session.get('user')).username:
                context = {
                    'post':post,
                    'update_auth':"ok",
                    "num":num,
                }
            else:
                context = {
                    'post':post,
                    "num":num,
                }
            return render(request, 'board/detail.html', context)
    except:
        return render(request, 'board/error.html')


@login_required
def write(request):
    if request.method == 'POST':
        post = Post.objects.create(title = request.POST.get('title'),
                                   body = request.POST.get('body'),
                                   photo = request.FILES.get('img'),
                                   writer = User.objects.get(username = request.POST.get('writer')))
        post.save()
        
        nums = len(Post.objects.all())
        return redirect('board:detail', id=post.id ,num=1)
    
    user_id = request.session.get('user')
    user = User.objects.get(username=user_id)
    return render(request, 'board/write.html', {"writer":user.username})

@auth_required
@login_required
def update(request,id):
    try:
        post = Post.objects.get(id=id)
        if post.is_active:
            if request.method == 'POST':
                post.title = request.POST.get('title')
                post.body = request.POST.get('body') 
                photo = request.FILES
                if photo:
                    post.photo = photo.get('img')
                post.save()
                nums = len(Post.objects.all())
                return redirect('board:detail', id=id ,num=nums)
            return render(request, 'board/update.html',{"post":post})
    except:
        return render(request, 'board/error.html')


@auth_required
@login_required
def delete(request,id):
    try:
        post = Post.objects.get(id=id)
        if post.is_active:
            post.is_active=False
            post.save()
            return redirect('board:index', page=1)
    except:
        return render(request, 'board/error.html')

