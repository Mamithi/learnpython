from django.shortcuts import render
from . import forms
from send_email.settings import EMAIL_HOST_USER
from django.core.mail import send_mail
from django.shortcuts import render


# Create your views here.
def subscribe(request):
    sub = forms.Subscribe()
    if request.method == 'POST':
        sub = forms.Subscribe(request.POST)
        subject = 'Welcome Home'
        message = 'This is a message for testing'
        recipient = str(sub['Email'].value())
        send_mail(subject, message, EMAIL_HOST_USER, [recipient], fail_silently=False)
        return render(request, 'subscribe/success.html', {'recipient': recipient})
    return render(request, 'subscribe/index.html', {'form': sub})
