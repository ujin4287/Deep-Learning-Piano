import random

left_hand = random.randrange(1, 14)
right_hand = random.randrange(1, 14)

finger_left = 0
finger_right = 0

print("left hand:", left_hand)
print("right hand:", right_hand)

if left_hand > 4 and right_hand > 4:
    finger_left = random.randrange(0, 5)
    finger_right = random.randrange(0, 5)

if left_hand <= 4:
    finger_left = random.randrange(0, left_hand)

if right_hand <= 4:
    finger_right = random.randrange(0, right_hand)

print("left_finger:",finger_left)
print("right_finger:",finger_right)

left_list_temp=[]
right_list_temp=[]

for i in range(1, left_hand):
    left_list_temp.append(i)

for i in range(1, right_hand):
    right_list_temp.append(i)

print(left_list_temp)
print(right_list_temp)

print("random value:")

left_out = random.sample(left_list_temp, finger_left)
right_out = random.sample(right_list_temp, finger_right)

print(left_out)
print(right_out)

left_out.sort()
right_out.sort()

print(left_out)
print(right_out)


if (not left_out) and (not right_out):
    print("empty")


# 왼손 시작
temp1 = 36 - left_hand-right_hand # 0번째 배열부터 이다.
left_start = random.randrange(0, temp1)

# 오른손 시작
temp2 = left_start + left_hand
right_start = random.randrange(temp2, 36-right_hand)

print("left : start - end")
print(left_start, left_start + left_hand - 1)
print("right : start - end:")
print(right_start, right_start + right_hand - 1)

final_list = []

print("left_position")
for i in left_out:
    print(left_start + i)
    final_list.append(left_start + i)

print("right_position")
for i in right_out:
    print(right_start + i)
    final_list.append(right_start + i)

print(final_list)

