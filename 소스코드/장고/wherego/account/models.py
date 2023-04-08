from django.db import models
# Create your models here.
from .generateroom import makeroom

class User(models.Model):
    username = models.CharField('계정',
                            max_length=10,
                            unique=True,
                            help_text='필수. 150자 이하. 문자, 숫자만 사용',
                            error_messages={
                                'unique': "존재하는 계정입니다",
                            },
                        )
    password = models.CharField('비밀번호', max_length=20)
    name = models.CharField('이름', max_length=150)
    email = models.EmailField('이메일', unique=True)
    gender = models.CharField('성별', max_length=1)
    birthday= models.CharField('생년월일', max_length=10)
    is_active = models.BooleanField(default=False, help_text='계정을 삭제하는 대신 선택을 해제함')
    date_joined = models.DateTimeField('가입일자', auto_now_add=True)

    def __str__(self):
        return self.username
    
class Group(models.Model):
    id = id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=128)
    members = models.ManyToManyField(User)
    chattingid = models.CharField(max_length=128, default=makeroom(),unique=True)
    
    def __str__(self):
        return self.name 