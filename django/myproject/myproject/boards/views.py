from django.shortcuts import render, get_object_or_404
from django.http import Http404
from .models import Board

# Create your views here.


def home(request):
    boards = Board.objects.all()

    return render(request, 'home.html', {'boards': boards})


def board_topics(request, pk):
    board = get_object_or_404(Board, pk=pk)

    return render(request, 'topics.html', {'board': board})

def new_topic(request, pk):
    board = get_object_or_404(Board, pk)
    return render(request, 'new_topic.html', {'board': board})
