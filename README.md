This is the repository for our GNN project.

these are the steps you need to take to start working with me:
1. install git if you haven't already, here's a link
https://git-scm.com/download/win
2. run the following commands in cmd:
git config --global user.name "Jonathan Kouchly"
git config --global user.email "nonkiponk@gmail.com"
3. open the cmd and move to the GNNs directory, then clone this repo, run:
git clone 
it might ask you to authenticate, if it does then your github password isn't gonna work. You'll have to generate a token on github and use it instead of the password. In order to generate a token you need to go to settings->developer settings->something token something. Make sure you copy it because you're only gonna see the token once.
4. start working and don't forget to commit and push whenever you're done
and to pull whenever you start. The commands are supposed to look like this:
"git pull origin main"
"git push -u origin main"
5. (optional) take a look at this tutorial on how to use git i think it's really important: https://www.w3schools.com/git/default.asp?remote=github


If you want to pull changes that i did on files that you work on as well, then follow those steps:
git fetch origin,
git reset --hard origin/main,
git pull origin main
