while True:
    print("Selection: Rock | Paper | Scissors")

    player_one = input("Player one selects: ")
    player_two = input("Player two selects: ")

    if player_one == player_two:
        print("It is a tie!")
    elif ((player_one == "Rock" and player_two == "Scissors" ) or (player_one == "Paper" and player_two == "Rock") or (player_one == "Scissors" and player_two == "Paper")):
        print("Player One wins")
    elif ((player_two == "Rock" and player_one == "Scissors" ) or (player_two == "Paper" and player_one == "Rock") or (player_two == "Scissors" and player_one == "Paper")):
        print("Player Two wins")