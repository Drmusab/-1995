import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
} from 'n8n-workflow';

export class AiAssistantChat implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'AI Assistant Chat',
		name: 'aiAssistantChat',
		icon: 'file:aiassistant.svg',
		group: ['ai'],
		version: 1,
		subtitle: '={{$parameter["operation"] + ": " + $parameter["resource"]}}',
		description: 'Interact with the AI Assistant for chat conversations',
		defaults: {
			name: 'AI Assistant Chat',
		},
		inputs: ['main'],
		outputs: ['main'],
		credentials: [
			{
				name: 'aiAssistantApi',
				required: true,
			},
		],
		requestDefaults: {
			baseURL: '={{$credentials.baseUrl}}/api/v1',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
			},
		},
		properties: [
			{
				displayName: 'Resource',
				name: 'resource',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'Chat',
						value: 'chat',
					},
					{
						name: 'Session',
						value: 'session',
					},
				],
				default: 'chat',
			},
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				displayOptions: {
					show: {
						resource: ['chat'],
					},
				},
				options: [
					{
						name: 'Send Message',
						value: 'sendMessage',
						description: 'Send a message to the AI Assistant',
						action: 'Send a message',
					},
					{
						name: 'Get Chat History',
						value: 'getChatHistory',
						description: 'Get chat history for a session',
						action: 'Get chat history',
					},
				],
				default: 'sendMessage',
			},
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				displayOptions: {
					show: {
						resource: ['session'],
					},
				},
				options: [
					{
						name: 'Create Session',
						value: 'createSession',
						description: 'Create a new chat session',
						action: 'Create a session',
					},
					{
						name: 'Get Session',
						value: 'getSession',
						description: 'Get session information',
						action: 'Get a session',
					},
					{
						name: 'Delete Session',
						value: 'deleteSession',
						description: 'Delete a chat session',
						action: 'Delete a session',
					},
				],
				default: 'createSession',
			},
			// Chat Operations
			{
				displayName: 'Message',
				name: 'message',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['sendMessage'],
						resource: ['chat'],
					},
				},
				default: '',
				placeholder: 'Enter your message here',
				description: 'The message to send to the AI Assistant',
			},
			{
				displayName: 'Session ID',
				name: 'sessionId',
				type: 'string',
				displayOptions: {
					show: {
						operation: ['sendMessage', 'getChatHistory'],
						resource: ['chat'],
					},
				},
				default: '',
				placeholder: 'Optional: Session ID for context',
				description: 'Session ID to maintain conversation context. If not provided, a new session will be created.',
			},
			{
				displayName: 'User ID',
				name: 'userId',
				type: 'string',
				displayOptions: {
					show: {
						operation: ['sendMessage'],
						resource: ['chat'],
					},
				},
				default: '',
				placeholder: 'Optional: User identifier',
				description: 'User identifier for personalized responses',
			},
			{
				displayName: 'Context',
				name: 'context',
				type: 'json',
				displayOptions: {
					show: {
						operation: ['sendMessage'],
						resource: ['chat'],
					},
				},
				default: '{}',
				description: 'Additional context data as JSON object',
			},
			// Session Operations
			{
				displayName: 'Session ID',
				name: 'sessionId',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['getSession', 'deleteSession'],
						resource: ['session'],
					},
				},
				default: '',
				description: 'The session ID to operate on',
			},
			{
				displayName: 'User ID',
				name: 'userId',
				type: 'string',
				displayOptions: {
					show: {
						operation: ['createSession'],
						resource: ['session'],
					},
				},
				default: '',
				description: 'User ID for the new session',
			},
			{
				displayName: 'Session Type',
				name: 'sessionType',
				type: 'options',
				displayOptions: {
					show: {
						operation: ['createSession'],
						resource: ['session'],
					},
				},
				options: [
					{
						name: 'Interactive',
						value: 'interactive',
					},
					{
						name: 'Batch',
						value: 'batch',
					},
					{
						name: 'API',
						value: 'api',
					},
				],
				default: 'interactive',
				description: 'Type of session to create',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];
		const credentials = await this.getCredentials('aiAssistantApi');

		for (let i = 0; i < items.length; i++) {
			const resource = this.getNodeParameter('resource', i) as string;
			const operation = this.getNodeParameter('operation', i) as string;

			let responseData;

			try {
				if (resource === 'chat') {
					if (operation === 'sendMessage') {
						const message = this.getNodeParameter('message', i) as string;
						const sessionId = this.getNodeParameter('sessionId', i) as string;
						const userId = this.getNodeParameter('userId', i) as string;
						const context = this.getNodeParameter('context', i) as string;

						let contextData = {};
						if (context) {
							try {
								contextData = JSON.parse(context);
							} catch (error) {
								throw new NodeOperationError(this.getNode(), 'Invalid JSON in context field');
							}
						}

						const body = {
							message,
							session_id: sessionId || undefined,
							user_id: userId || undefined,
							context: contextData,
						};

						responseData = await this.helpers.requestWithAuthentication.call(
							this,
							'aiAssistantApi',
							{
								method: 'POST',
								url: '/chat',
								body,
								json: true,
							},
						);
					} else if (operation === 'getChatHistory') {
						const sessionId = this.getNodeParameter('sessionId', i) as string;

						responseData = await this.helpers.requestWithAuthentication.call(
							this,
							'aiAssistantApi',
							{
								method: 'GET',
								url: `/sessions/${sessionId}/history`,
								json: true,
							},
						);
					}
				} else if (resource === 'session') {
					if (operation === 'createSession') {
						const userId = this.getNodeParameter('userId', i) as string;
						const sessionType = this.getNodeParameter('sessionType', i) as string;

						const body = {
							user_id: userId || undefined,
							session_type: sessionType,
						};

						responseData = await this.helpers.requestWithAuthentication.call(
							this,
							'aiAssistantApi',
							{
								method: 'POST',
								url: '/sessions',
								body,
								json: true,
							},
						);
					} else if (operation === 'getSession') {
						const sessionId = this.getNodeParameter('sessionId', i) as string;

						responseData = await this.helpers.requestWithAuthentication.call(
							this,
							'aiAssistantApi',
							{
								method: 'GET',
								url: `/sessions/${sessionId}`,
								json: true,
							},
						);
					} else if (operation === 'deleteSession') {
						const sessionId = this.getNodeParameter('sessionId', i) as string;

						responseData = await this.helpers.requestWithAuthentication.call(
							this,
							'aiAssistantApi',
							{
								method: 'DELETE',
								url: `/sessions/${sessionId}`,
								json: true,
							},
						);
					}
				}

				const executionData = this.helpers.constructExecutionMetaData(
					this.helpers.returnJsonArray(responseData as INodeExecutionData),
					{ itemData: { item: i } },
				);

				returnData.push(...executionData);
			} catch (error) {
				if (this.continueOnFail()) {
					const executionData = this.helpers.constructExecutionMetaData(
						this.helpers.returnJsonArray({ error: error.message }),
						{ itemData: { item: i } },
					);
					returnData.push(...executionData);
					continue;
				}
				throw error;
			}
		}

		return [returnData];
	}
}