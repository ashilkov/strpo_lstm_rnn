from rnn import RNN
import msvcrt

rnn = RNN()
# rnn.show_plots()

print('\nInput text')
line = ''
while(True):
    if msvcrt.kbhit():
        char = msvcrt.getch()
        key = ord(char)
        if line == 'exit':
            break

        if key == 13:
            line = ''
            print('\n')

        elif key == 8:
            line = line[:len(line) - 1]

        elif char:
            msvcrt.putch(char)
            line += chr(key)

        if len(line) > 3:
            prediction = rnn.predict(line.lower(),3)
            print('\n')
            print(prediction)
            [msvcrt.putwch(ch) for ch in line]