const express=require('express')
require("./db/mongoDB.js")
const path=require('path')
const userRouter=require("./router/users.js")
const app=express()

const port=process.env.PORT || 4000
const publicDirectoryPath=path.join(__dirname,'../public')

app.use(express.static(publicDirectoryPath))
app.use(express.json())
app.use(userRouter)


app.listen(port,()=>{
    console.log("server is up and running on ",port)
})