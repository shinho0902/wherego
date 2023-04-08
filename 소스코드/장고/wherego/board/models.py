from django.db import models
from account.models import User
# Create your models here.
class Post(models.Model):
    id = models.AutoField(primary_key=True)
    group = models.CharField('그룹', max_length=20, default='')
    title = models.CharField('제목', max_length=250)
    body = models.TextField('내용')
    region = models.CharField('장소',max_length=20, default='')
    photo = models.ImageField()
    writer =  models.ForeignKey(User, on_delete=models.CASCADE)
    views = models.IntegerField('조회수', default = 0)
    is_active = models.BooleanField(default=True, help_text='계정을 삭제하는 대신 선택을 해제함')

    created = models.DateField(auto_now_add=True) 
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
