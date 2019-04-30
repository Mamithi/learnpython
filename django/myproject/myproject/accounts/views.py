from django.shortcuts import render
from .forms import SignUpForm
from django.contrib.auth import login as auth_login
from django.shortcuts import render, redirect

# Create your views here.
def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            return redirect('home')
    else:
        form = SignUpForm()    
    return render(request, 'signup.html', {'form': form})