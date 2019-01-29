function withdraw()
uint transferAmt = 1 ether; 
if (!msg.sender.call.value(transferAmt)()) throw; 
