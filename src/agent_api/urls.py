from django.urls import path

from agent_api import views

urlpatterns = [
    path("api/ask", views.ask, name="ask"),
]