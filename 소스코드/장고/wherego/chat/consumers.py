import json

from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from account.models import User,Group
from .models import Message

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = await database_sync_to_async(self.get_name)()
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name = "chat_%s" % self.room_name

        # Join room group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)

        await self.accept()

    def get_name(self):
        return User.objects.get(username = self.scope['session']['user'])

    @database_sync_to_async
    def create_chat(self, msg, sender):
        Message.objects.create(user=sender, message=msg, room=Group.objects.get(chattingid=self.room_name))

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    # Receive message from WebSocket
    async def receive(self, text_data):
        username = self.user.username
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]
        message = (username+": "+message)
        new_msg = await self.create_chat(str(message.split(":")[1][1:]),await database_sync_to_async(self.get_name_by_name)(username))

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name, {"type": "chat_message", "message": message, "sender":username}
        )

    # Receive message from room group
    async def chat_message(self, event):
        message = event["message"]
        username = event["sender"]
        # Send message to WebSocket
        await self.send(text_data=json.dumps({"message": message,
        'username':username}))


    def get_name_by_name(self,username):
        return User.objects.get(username=username)