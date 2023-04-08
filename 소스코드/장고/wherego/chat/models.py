from django.db import models
from account.models import User,Group
# Create your models here.

class Message(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE) #(2)
    room = models.ForeignKey(Group, on_delete=models.CASCADE) #(3)
    date = models.DateTimeField(auto_now_add=True)
    message = models.CharField(max_length=5000)